# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import collections.abc
import contextlib
import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import deepspeed
import torch
import torch.distributed
import torch.utils.data
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric

from composer.core import Callback, Engine, Event, Logger, State
from composer.core.algorithm import Algorithm
from composer.core.logging import BaseLoggerBackend, LogLevel
from composer.core.types import Batch, BreakEpochException, Metrics, Precision, Tensor
from composer.datasets import DataloaderHparams, DataloaderSpec
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.models.base import BaseMosaicModel
from composer.optim import (ComposedScheduler, CosineAnnealingLRHparams, DecoupledSGDWHparams, OptimizerHparams,
                            SchedulerHparams, WarmUpLRHparams)
from composer.optim.scheduler import ensure_warmup_last
from composer.trainer.checkpoint import Checkpointer, CheckpointLoader
from composer.trainer.ddp import DDP, DataloaderMultipleIterationWarning
from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.scaler import ClosureGradScaler
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import ensure_tuple, get_random_seed, map_collection, seed_all

log = logging.getLogger(__name__)


class DeepSpeedTrainer:
    """Trainer for training a model with algorithms.

    Can be created either with ``__init__`` or by providing a
    :class:`~composer.trainer.TrainerHparams` object
    (see :meth:`~composer.trainer.Trainer.create_from_hparams`).

    Args:
        model (BaseMosaicModel): The model to train.
        train_dataloader_spec (DataloaderSpec): The dataloader spec for the training data.
        eval_dataloader_spec (DataloaderSpec): The dataloader spec for the evaluation data.
        max_epochs (int): The maximum number of epochs to train for.
        train_batch_size (int): Minibatch size for training data.
        eval_batch_size (int): Minibatch size for evaluation data.
        algorithms (List[Algorithm], optional): The algorithms to use during training.
            (default: ``[]``)
        optimizer_hparams: (OptimizerHparams, optional): The OptimizerHparams for constructing
            the optimizer for training. Must pass OptimizerHparams instead of a `torch.optim.Optimizer`
            object because the optimizer has to be constructed after certain algorithms which modify
            the model architecture have run on the model. (default:
            ``MosaicMLSGDWHparams(lr=0.1, momentum=0.9, weight_decay=1.0e-4)``)
        schedulers_hparams: (Union[SchedulerHparams, List[SchedulerHparams]], optional): The
            SchedulerHparams for constructing the one or more learning rate schedulers used
            during training. Must pass SchedulerHparams instead of a `torch.optim.lr_scheduler._LRScheduler`
            object because the scheduler needs an optimizer to be constructed and we construct the optimizer
            in `__init__`. (default:
            ``[CosineAnnealingLRHparams(T_max=f"{max_epochs}ep"), WarmUpLRHparams()]``).
        grad_accum (int, optional): The number of microbatches to split a per-device batch into. Gradients
            are summed over the microbatches per device. (default: ``1``)
        grad_clip_norm (float, optional): The norm to clip gradient magnitudes to. Set to None for no gradient
            clipping. (default: ``None``)
        validate_every_n_batches (int, optional): Compute metrics on evaluation data every N batches.
             Set to -1 to never validate on a batchwise frequency. (default: ``-1``)
        validate_every_n_epochs (int, optional): Compute metrics on evaluation data every N epochs.
            Set to -1 to never validate on a epochwise frequency. (default: ``1``)
        compute_training_metrics (bool, optional): True to compute metrics on training data and False to not.
            (default: ``False``)
        precision (Precision, optional): Numerical precision to use for training. (default: ``Precision.FP32``).
        num_workers (int, optional): The number of CPU workers to use per GPU. 0 results in loading data
            on the main process. (default: ``0``)
        prefetch_factor (int, optional): Number of samples loaded in advance by each worker. (default: ``2``)
        persistent_workers (bool, optional): Whether or not to shutdown workers after the dataset
            has been consumed once. (default: ``False``)
        pin_memory (bool, optional): Whether or not to copy data tensors into CUDA pinned memory.
            (default: ``False``)
        timeout (int, optional): Timeout value for collecting a batch from workers. 0 for no timeout.
            (default: ``0``)
        seed (int, optional): The seed used in randomization. When not provided a random seed
            will be created. (default: ``None``)
        log_destinations (List[BaseLoggerBackend], optional): The destinations to log training information to.
            (default ``[TQDMLoggerBackend()]``).
        callbacks (Sequence[Callback], optional): The callbacks to run during training. (default: ``[]``)
        checkpoint_filepath (str, optional): The path to a trainer checkpoint file. If provided
            the trainer will load the state (along with it's associated attributes) during initialization.
            (default: ``None``)
        checkpoint_interval_unit (int, optional): Unit for the checkpoint save interval -- should be 'ep'
            for epochs, 'ba' for batches, or None to disable checkpointing. (default: ``None``).
        checkpoint_folder (str, optional): The folder to save checkpoints to. (default: ``checkpoints``)
        checkpoint_interval (int, optional): The frequency with which to checkpoint. (default: ``1``)
        config (Dict[str, Any], optional): Extra user-provided trainer configuration. Will be persisted
            along with the trainer state during checkpointing. (default: ``None``)

    Attributes:
        state (State): The :class:`State` object used to store training state.
        logger (Logger): The :class:`Logger` used for logging.
        engine (Engine): The :class:`Engine` used for running callbacks and algorithms.
    """

    def __init__(
            self,
            *,
            model: BaseMosaicModel,
            train_dataloader_spec: DataloaderSpec,
            eval_dataloader_spec: DataloaderSpec,
            max_epochs: int,
            train_batch_size: int,
            eval_batch_size: int,
            algorithms: Optional[List[Algorithm]] = None,
            optimizer_hparams: Optional[OptimizerHparams] = None,
            schedulers_hparams: Optional[Union[SchedulerHparams, List[SchedulerHparams]]] = None,

            # training hparams
            grad_accum: int = 1,
            grad_clip_norm: Optional[float] = None,
            validate_every_n_batches: int = -1,
            validate_every_n_epochs: int = 1,
            compute_training_metrics: bool = False,
            precision: Precision = Precision.FP32,

            # dataloader hparams
            num_workers: int = 0,
            prefetch_factor: int = 2,
            persistent_workers: bool = False,
            pin_memory: bool = False,
            timeout: int = 0,

            # Randomness
            seed: Optional[int] = None,

            # Logging and callbacks
            log_destinations: Optional[List[BaseLoggerBackend]] = None,
            callbacks: Sequence[Callback] = tuple(),

            # Checkpoint hparams
            checkpoint_filepath: Optional[str] = None,
            checkpoint_interval_unit: Optional[str] = None,
            checkpoint_folder: Optional[str] = "checkpoints",
            checkpoint_interval: Optional[int] = 1,

            # Optional config (ex. an hparams yaml file)
            config: Optional[Dict[str, Any]] = None):
        # surpressing GradScaler warnings as they are always created
        # self._use_grad_scaling() will raise a RuntimeError if grad scaling is not available when it is required
        warnings.filterwarnings(action="ignore", message="torch.cuda.amp.GradScaler")

        self.config = config

        self.device = DeviceGPU(prefetch_in_cuda_stream=False)

        if not seed:
            # Set a deterministic seed in the hparams
            # This seed will be dumped in the hparams that are saved with checkpoints
            seed = get_random_seed()
            log.info(f"Seed was None. Setting seed to random value: {seed}")
        # If hparams is used to create the Trainer this function is called twice
        # which is okay because all runs with the hparams codepath will do this
        seed_all(seed)
        self.seed = seed

        if not algorithms:
            algorithms = []

        self.backwards_create_graph = any(map(lambda x: x.backwards_create_graph, algorithms))

        self.state = State(max_epochs=max_epochs,
                           train_batch_size=train_batch_size,
                           eval_batch_size=eval_batch_size,
                           algorithms=algorithms,
                           callbacks=callbacks,
                           model=model,
                           grad_accum=grad_accum,
                           precision=precision)

        if not log_destinations:
            log_destinations = [TQDMLoggerBackend()]
        self.logger = Logger(self.state, log_destinations)
        self.state.callbacks = [*log_destinations, *callbacks]

        self.engine = Engine(self.state, self.state.algorithms, self.logger, self.state.callbacks)

        self.train_dl_spec = train_dataloader_spec
        self.eval_dl_spec = eval_dataloader_spec

        self.dl_hparams = DataloaderHparams(num_workers=num_workers,
                                            prefetch_factor=prefetch_factor,
                                            persistent_workers=persistent_workers,
                                            pin_memory=pin_memory,
                                            timeout=timeout)

        self.validate_every_n_batches = validate_every_n_batches
        self.validate_every_n_epochs = validate_every_n_epochs
        self.compute_training_metrics = compute_training_metrics
        self.grad_clip_norm = grad_clip_norm

        # run INIT event before optimizers and schedulers are created
        self.engine.run_event(Event.INIT)

        assert isinstance(self.train_dl_spec.dataset, collections.abc.Sized)
        steps_per_epoch = len(self.train_dl_spec.dataset) // train_batch_size
        # Need to use hparams here because optimizer and schedulers need to be created after Event.INIT
        if not optimizer_hparams:
            optimizer_hparams = DecoupledSGDWHparams(lr=0.1, momentum=0.9, weight_decay=1.0e-4)
        if not schedulers_hparams:
            schedulers_hparams = [CosineAnnealingLRHparams(T_max=f"{max_epochs}ep"), WarmUpLRHparams()]
        if not isinstance(schedulers_hparams, list):
            schedulers_hparams = [schedulers_hparams]
        optimizer = optimizer_hparams.initialize_object(param_group=self.state.model.parameters())
        schedulers = [x.initialize_object(optimizer, steps_per_epoch) for x in ensure_warmup_last(schedulers_hparams)]
        self.state.optimizers = optimizer
        self.state.schedulers = ComposedScheduler(schedulers=schedulers)

        self.checkpointer = None
        if checkpoint_folder and checkpoint_interval and checkpoint_interval_unit:
            self.checkpointer = Checkpointer(checkpoint_folder=checkpoint_folder,
                                             checkpoint_interval=checkpoint_interval,
                                             checkpoint_interval_unit=checkpoint_interval_unit)

        self.checkpoint_loader = None
        if checkpoint_filepath:
            self.checkpoint_loader = CheckpointLoader(checkpoint_filepath=checkpoint_filepath)
            self.checkpoint_loader.load_checkpoint(state=self.state)

    @classmethod
    def create_from_hparams(cls, hparams: TrainerHparams) -> DeepSpeedTrainer:
        """Instantiate a Trainer using a `TrainerHparams` object.

        Args:
            hparams (TrainerHparams): The TrainerHparams object used to instantiate the trainer.

        Returns:
            A Trainer object initialized with the provided TrainerHparams.
        """

        hparams.validate()

        seed = hparams.seed if hparams.seed else get_random_seed()
        # need to set seed before model initialization for determinism
        seed_all(seed)

        model = hparams.model.initialize_object()
        algorithms = [x.initialize_object() for x in hparams.algorithms]

        # callbacks, loggers, and seed
        callbacks = [x.initialize_object() for x in hparams.callbacks]
        dict_config = hparams.to_dict()
        log_destinations = [x.initialize_object(config=dict_config) for x in hparams.loggers]

        train_dl_spec = hparams.train_dataset.initialize_object()
        eval_dl_spec = hparams.val_dataset.initialize_object()

        trainer = cls(
            model=model,
            train_dataloader_spec=train_dl_spec,
            eval_dataloader_spec=eval_dl_spec,
            max_epochs=hparams.max_epochs,
            train_batch_size=hparams.total_batch_size,
            eval_batch_size=hparams.eval_batch_size,
            algorithms=algorithms,
            optimizer_hparams=hparams.optimizer,
            schedulers_hparams=hparams.schedulers,

            # training hparams
            grad_accum=hparams.grad_accum,
            grad_clip_norm=hparams.grad_clip_norm,
            validate_every_n_batches=hparams.validate_every_n_batches,
            validate_every_n_epochs=hparams.validate_every_n_epochs,
            compute_training_metrics=hparams.compute_training_metrics,
            precision=hparams.precision,

            # dataloader hparams
            num_workers=hparams.dataloader.num_workers,
            prefetch_factor=hparams.dataloader.prefetch_factor,
            persistent_workers=hparams.dataloader.persistent_workers,
            pin_memory=hparams.dataloader.pin_memory,
            timeout=hparams.dataloader.timeout,

            # Randomness
            seed=seed,

            # Callbacks and logging
            log_destinations=log_destinations,
            callbacks=tuple(callbacks),

            # Checkpointing hparams
            checkpoint_filepath=hparams.checkpoint_filepath,
            checkpoint_interval_unit=hparams.checkpoint_interval_unit,
            checkpoint_folder=hparams.checkpoint_folder,
            checkpoint_interval=hparams.checkpoint_interval,

            # Optional config
            config=hparams.to_dict())

        return trainer

    def fit(self):
        """Train and evaluate the model on the provided data."""
        self._train_loop()

    def _create_dataloaders(self) -> None:
        """Create the dataloaders.

        Should be called after distributed training has started,
        since the dataloader samplers need to know their rank.
        """
        # shorthand
        state = self.state

        # compute per gpu batch size
        train_gpu_batch_size = state.train_batch_size // state.world_size
        eval_gpu_batch_size = state.eval_batch_size // state.world_size

        train_dataloader = self.dl_hparams.initialize_object(batch_size=train_gpu_batch_size,
                                                             sampler=torch.utils.data.DistributedSampler[int](
                                                                 self.train_dl_spec.dataset,
                                                                 drop_last=self.train_dl_spec.drop_last,
                                                                 shuffle=self.train_dl_spec.shuffle),
                                                             dataloader_spec=self.train_dl_spec)

        eval_dataloader = self.dl_hparams.initialize_object(batch_size=eval_gpu_batch_size,
                                                            sampler=torch.utils.data.DistributedSampler[int](
                                                                self.eval_dl_spec.dataset,
                                                                drop_last=self.eval_dl_spec.drop_last,
                                                                shuffle=self.eval_dl_spec.shuffle),
                                                            dataloader_spec=self.train_dl_spec)

        # move to device
        state.train_dataloader = self.device.dataloader_to_device(
            train_dataloader,
            self.train_dl_spec.prefetch_fn,
        )
        state.eval_dataloader = self.device.dataloader_to_device(
            eval_dataloader,
            self.eval_dl_spec.prefetch_fn,
        )

    def _get_metrics_as_collection(self, *, is_train: bool) -> MetricCollection:
        """Get metrics relevant to the model. Metrics are all implemented as subclasses
        of :class:`torchmetrics.Metric`. This function returns metrics as a
        :class:`~torchmetrics.collections.MetricCollection` to enable support
        for multiple metrics.

        Args:
            is_train (bool): True to get training metrics and false to get
            evaluation metrics.

        Returns:
            A :class:`~torchmetrics.collections.MetricCollection` object.
        """
        assert isinstance(self.state.model, BaseMosaicModel)

        metrics = self.state.model.metrics(train=is_train)
        assert isinstance(metrics, (Metric, MetricCollection)), \
            "Error module.metrics() must return a Metric or MetricCollection object."
        if isinstance(metrics, Metric):
            # Forcing metrics to be a MetricCollection simplifies logging results
            metrics = MetricCollection([metrics])

        # Safety check to ensure the metric and data are on the same device. Normally not
        # needed because the metric is automatically on the same device as the model.
        # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html for details.
        metrics = self.device.module_to_device(metrics)

        # HACK: DeepSpeed somehow manages to convert metric internal states to its own dtype. When
        # running with FP16, this tends to result in overflows. Let's assume FP32 is good enough.
        for _, metric in metrics.items():
            metric.set_dtype(torch.float32)

        return metrics

    def _compute_and_log_metrics(self, metrics: Metrics, *, is_train: bool, is_batch: bool):
        """Computes metrics, logs the results, and resets the metrics.

        Args:
            metrics (Metrics): The metrics to compute.
            is_train (bool): True for training metrics, False for evaluation metrics.
            is_batch (bool): True if logging at batch level, false for epoch level.
        """
        computed_metrics = metrics.compute()
        for name, value in computed_metrics.items():
            log_level = LogLevel.BATCH if is_batch else LogLevel.EPOCH
            suffix = 'train' if is_train else 'val'
            self.logger.metric(log_level, {f'{name.lower()}/{suffix}': value})
        metrics.reset()

    def _spin_dataloaders(self):
        """Spin the dataloaders to restore sampler state.

        Only one batch must be loaded to seed the sampler's generator.
        since only the first batch is being loaded, the dataloader may
        not be completely iterated through.
        """
        # surpressing this multiple iteration warning -- it is OK to ignore
        warnings.simplefilter(action="ignore", category=DataloaderMultipleIterationWarning, append=True)
        assert self.state.train_dataloader is not None, "train dataloader should be set"
        assert self.state.eval_dataloader is not None, "eval dataloader should be set"

        # spin the eval dataloader once to initialize its sampler deterministically
        # so it does not affect any other RNG reads
        for _ in self.state.eval_dataloader:
            break

        # spin the train dataloader's sampler to get to the state of the desired epoch
        for _ in range(self.state.epoch):
            for _ in self.state.train_dataloader:
                break

    def _get_batch_size(self, batch: Batch) -> int:
        if isinstance(batch, Tensor):
            return batch.shape[0]

        dim0_sizes = []
        if isinstance(batch, (list, tuple)):
            for tensors in batch:
                for t in ensure_tuple(tensors):
                    dim0_sizes.append(t.shape[0])
        elif isinstance(batch, dict):
            dim0_sizes = [t.shape[0] for t in batch.values()]

        if len(set(dim0_sizes)) == 1:
            return dim0_sizes[0]
        else:
            raise ValueError('The default _get_batch_size function found ',
                             f'multiple Tensor sizes in batch: {dim0_sizes}')

    def _train_loop(self) -> None:
        """Run training for the specified number of epochs and log results."""
        # shorthand
        state = self.state

        assert state.optimizers is not None

        assert len(ensure_tuple(state.optimizers)) == 1
        optimizer = ensure_tuple(state.optimizers)[0]

        deepspeed_config = {
            "train_batch_size": state.train_batch_size,
            "gradient_accumulation_steps": state.grad_accum,
        }

        if state.precision == Precision.AMP:
            deepspeed_config["amp"] = {"enabled": True}
        elif state.precision == Precision.FP16:
            deepspeed_config["fp16"] = {"enabled": True}

        if self.grad_clip_norm:
            deepspeed_config["gradient_clipping"] = self.grad_clip_norm

        (self.deepspeed_engine, state.optimizers, _, _) = deepspeed.initialize(
            config=deepspeed_config,
            model=state.model,
            optimizer=optimizer,
        )

        # place the state, model in the proper devices
        self.device.prepare(state)

        self._create_dataloaders()
        if state.train_dataloader is None or state.eval_dataloader is None:
            raise ValueError('Dataloaders were not created properly, and are None.')

        # print training start
        self.logger.metric_fit({"trainer/algorithms": [str(algo) for algo in self.engine.algorithms]})

        if self.compute_training_metrics:
            log.warn('Computing model evaluation metrics during training.'
                     ' This doubles the number of forward passes and may lead'
                     ' to a throughput degradation.')
        train_metrics = self._get_metrics_as_collection(is_train=True)

        self.engine.run_event(Event.TRAINING_START)

        if len(ensure_tuple(state.optimizers)) != 1:
            raise NotImplementedError("The Mosaic trainer only supports one optimizer; "
                                      f"found {len(ensure_tuple(state.optimizers))} optimizers")

        self._spin_dataloaders()

        if self.state.batch_idx == 0 and self.checkpoint_loader:
            # only restore the rng state here if the step in the current epoch is zero.
            self.checkpoint_loader.restore_checkpoint_rng_state(self.state, self.device)

        for _ in range(state.epoch, state.max_epochs):
            try:
                state.model.train()

                if self.state.batch_idx == 0:
                    self.engine.run_event(Event.EPOCH_START)
                    self.logger.metric_epoch({"epoch": self.state.epoch})

                assert state.train_dataloader, "Train Dataloader must be set"
                for batch_idx, state.batch in enumerate(state.train_dataloader):

                    # if resuming, skip dataloader forward to the minibatch index
                    if batch_idx < self.state.batch_idx:
                        if self.checkpoint_loader:
                            self.checkpoint_loader.restore_checkpoint_rng_state(self.state, self.device)
                        continue

                    state.last_batch_size = self._get_batch_size(state.batch)

                    if self.compute_training_metrics:
                        # compute metrics on the training set
                        self.deepspeed_engine.eval()
                        with torch.no_grad():
                            eval_microbatches = self.train_dl_spec.split_fn(state.batch, state.grad_accum)
                            for eval_microbatch in eval_microbatches:
                                # TODO: Detect if self.run_event(Event.AFTER_DATALOADER) changes the training
                                # data and if so print a warning that metrics may return unexpected results
                                outputs, targets = self.state.model.validate(eval_microbatch)
                                train_metrics.update(outputs, targets)

                    self.deepspeed_engine.train()

                    self.engine.run_event(Event.AFTER_DATALOADER)

                    microbatches = self.train_dl_spec.split_fn(state.batch, state.grad_accum)

                    self.engine.run_event(Event.BATCH_START)
                    self.logger.metric_batch({
                        "trainer/global_step": self.state.step,
                        "trainer/batch_idx": self.state.batch_idx,
                    })
                    total_loss = self._train_batch(microbatches)

                    if total_loss is not None:
                        assert isinstance(total_loss, Tensor)
                        all_losses = self.deepspeed_engine.all_gather_scalar(total_loss, dp_group=None)
                        full_loss = sum(all_losses).cpu().item()
                        self.logger.metric_batch({'loss/train': full_loss / state.world_size})

                    if self.compute_training_metrics:
                        self._compute_and_log_metrics(train_metrics, is_train=True, is_batch=True)

                    self.engine.run_event(Event.BATCH_END)

                    state.schedulers.step(interval='batch')  # type: ignore

                    if self.validate_every_n_batches > 0 and (state.step + 1) % self.validate_every_n_batches == 0:
                        self.eval(is_batch=True)

                    state.step += 1
                    if self.checkpointer and self.checkpointer.should_checkpoint(state=state, event=Event.BATCH_END):
                        self.checkpointer.save_checkpoint(state=state,
                                                          seed=self.seed,
                                                          device=self.device,
                                                          ddp=self.ddp,
                                                          config=self.config)
            except BreakEpochException:
                log.info(f'Skipping the rest of Epoch {state.epoch}')

            state.schedulers.step(interval='epoch')  # type: ignore
            self.engine.run_event(Event.EPOCH_END)

            if self.validate_every_n_epochs > 0 and (state.epoch + 1) % self.validate_every_n_epochs == 0:
                self.eval(is_batch=False)

            state.epoch += 1

            if self.checkpointer and self.checkpointer.should_checkpoint(state=state, event=Event.EPOCH_END):
                self.checkpointer.save_checkpoint(state=state,
                                                  seed=self.seed,
                                                  device=self.device,
                                                  ddp=self.ddp,
                                                  config=self.config)

        self.engine.run_event(Event.TRAINING_END)

    def _train_batch(self, microbatches: Sequence[Batch]):
        """Iterate over microbatches and compute the loss that will be used to step
        the optimizer.

        Args:
            microbatches (Sequence[Batch]): The microbatches which make up the batch.
        """
        self.engine.run_event(Event.BEFORE_TRAIN_BATCH)

        state = self.state

        # tracker for gradient accumulation
        total_loss = self.device.tensor_to_device(torch.zeros(size=(1,)))

        current_batch_size = sum([self._get_batch_size(batch) for batch in microbatches])

        for state.batch in microbatches:

            current_microbatch_size = self._get_batch_size(state.batch)

            # forward pass
            self.engine.run_event(Event.BEFORE_FORWARD)

            state.outputs = self.deepspeed_engine.forward(state.batch)

            self.engine.run_event(Event.AFTER_FORWARD)

            # loss
            self.engine.run_event(Event.BEFORE_LOSS)

            state.loss = self.state.model.loss(state.outputs, state.batch)

            # Loss is added to losses with clone to not scale the loss for the step printout
            # Likely need to look into the performance impact
            for loss in ensure_tuple(state.loss):
                cloned_loss = loss.detach().clone()
                cloned_loss.mul_(current_microbatch_size / current_batch_size)
                total_loss += cloned_loss

            assert state.loss is not None
            self.engine.run_event(Event.AFTER_LOSS)

            # backward
            self.engine.run_event(Event.BEFORE_BACKWARD)

            self.deepspeed_engine.backward(state.loss)

            self.engine.run_event(Event.AFTER_BACKWARD)

            # This seems weird, but it's how DeepSpeed's step() likes to be called
            self.deepspeed_engine.step()

        self.engine.run_event(Event.AFTER_TRAIN_BATCH)

        return total_loss

    def eval(self, is_batch: bool):
        """Evaluate the model on the provided evaluation data and log
        appropriate metrics.

        Args:
            is_batch (bool): True to log metrics with ``LogLevel.BATCH``
                and False to log metrics with ``LogLevel.EPOCH``.
        """

        state = self.state
        model = state.model

        restore_model_train = model.training

        self.deepspeed_engine.eval()
        with torch.no_grad():

            self.engine.run_event(Event.EVAL_START)

            assert isinstance(model, BaseMosaicModel)

            metrics = self._get_metrics_as_collection(is_train=False)

            assert state.eval_dataloader is not None

            for state.batch in state.eval_dataloader:
                self.engine.run_event(Event.EVAL_BATCH_START)

                self.engine.run_event(Event.EVAL_BEFORE_FORWARD)
                state.outputs, targets = model.validate(state.batch)
                self.engine.run_event(Event.EVAL_AFTER_FORWARD)

                metrics.update(state.outputs, targets)

                self.engine.run_event(Event.EVAL_BATCH_END)

            self._compute_and_log_metrics(metrics, is_train=False, is_batch=is_batch)
            self.engine.run_event(Event.EVAL_END)

        if restore_model_train:
            self.deepspeed_engine.train()
