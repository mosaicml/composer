# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import collections.abc
import contextlib
import datetime
import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed
import torch.utils.data
from torch.backends import cudnn
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
from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.scaler import ClosureGradScaler
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import ddp, ensure_tuple, get_random_seed, map_collection, seed_all
from composer.utils.run_directory import get_relative_to_run_directory

log = logging.getLogger(__name__)


class Trainer:
    """Trainer for training a model with algorithms.

    Can be created either with ``__init__`` or by providing a
    :class:`~composer.trainer.TrainerHparams` object
    (see :meth:`~composer.trainer.Trainer.create_from_hparams`).

    Args:
        model (BaseMosaicModel): The model to train.
        train_dataloader_spec (DataloaderSpec): The dataloader spec for the training data.
        eval_dataloader_spec (DataloaderSpec): The dataloader spec for the evaluation data.
        max_epochs (int): The maxmimum number of epochs to train for.
        train_batch_size (int): Minibatch size for training data.
        eval_batc_size (int): Minibatch size for evaluation data.
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
        device (Device, optional): The device to use for training. Either `DeviceCPU` or `DeviceGPU`.
            (default ``DeviceCPU(n_cpus=1)``)
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
        ddp_sync_strategy (DDPSyncStrategy, optional): The strategy to use for synchronizing gradients.
            Leave unset to let the trainer auto-configure this.
        ddp_timeout (float, optional): Timeout, in seconds, for initializing the DDP process group.
            (default: ``5.0``)
        seed (int, optional): The seed used in randomization. When not provided a random seed
            will be created. (default: ``None``)
        deterministic_mode (bool, optional): Run the model deterministically. Experimental. Performance
            degradations expected. Certain Torch modules may not have deterministic implementations,
            which will result in a crash. (default: ``False``)
        log_destinations (List[BaseLoggerBackend], optional): The destinations to log training information to.
            (default ``[TQDMLoggerBackend()]``).
        callbacks (Sequence[Callback], optional): The callbacks to run during training. (default: ``[]``)
        checkpoint_filepath (str, optional): The path to a trainer checkpoint file. If provided
            the trainer will load the state (along with it's associated attributes) during initialization.
            (default: ``None``)
        checkpoint_interval_unit (int, optional): Unit for the checkpoint save interval -- should be 'ep'
            for epochs, 'ba' for batches, or None to disable checkpointing. (default: ``None``).
        checkpoint_folder (str, optional): The folder to save checkpoints to. Relative to `os.environ.get('RUN_DIRECTORY', '.')`, 
            (default: ``checkpoints``)
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

            # device
            device: Optional[Device] = None,

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

            # ddp hparams
            ddp_sync_strategy: Optional[Union[str, ddp.DDPSyncStrategy]] = None,
            ddp_timeout: float = 5.0,

            # Randomness
            seed: Optional[int] = None,
            deterministic_mode: bool = False,

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

        if not device:
            device = DeviceCPU()
        self.device = device
        self.device.prepare()

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

        find_unused_parameters = any(map(lambda x: x.find_unused_parameters, algorithms))

        self.find_unused_parameters = find_unused_parameters
        if ddp_sync_strategy is None:
            self.ddp_sync_strategy = ddp.DDPSyncStrategy.SINGLE_AUTO_SYNC if not find_unused_parameters else ddp.DDPSyncStrategy.FORCED_SYNC
        else:
            self.ddp_sync_strategy = ddp.DDPSyncStrategy(ddp_sync_strategy)

        ddp.initialize_ddp(device.ddp_backend, datetime.timedelta(seconds=ddp_timeout))

        dl_hparams = DataloaderHparams(num_workers=num_workers,
                                       prefetch_factor=prefetch_factor,
                                       persistent_workers=persistent_workers,
                                       pin_memory=pin_memory,
                                       timeout=timeout)

        train_gpu_batch_size = train_batch_size // ddp.get_world_size()
        train_dataloader = self.device.dataloader_to_device(
            ddp.create_dataloader(train_gpu_batch_size, dl_hparams, train_dataloader_spec),
            train_dataloader_spec.prefetch_fn,
        )
        self.train_dl_spec = train_dataloader_spec

        eval_gpu_batch_size = eval_batch_size // ddp.get_world_size()
        eval_dataloader = self.device.dataloader_to_device(
            ddp.create_dataloader(eval_gpu_batch_size, dl_hparams, eval_dataloader_spec),
            eval_dataloader_spec.prefetch_fn,
        )
        self.eval_dl_spec = eval_dataloader_spec

        self.state = State(
            max_epochs=max_epochs,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            algorithms=algorithms,
            callbacks=callbacks,
            model=model,
            grad_accum=grad_accum,
            precision=precision,
            precision_context=self.device.precision_context,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

        if not log_destinations:
            log_destinations = [TQDMLoggerBackend()]
        self.logger = Logger(self.state, log_destinations)
        self.state.callbacks = [*log_destinations, *callbacks]

        self.engine = Engine(self.state, self.state.algorithms, self.logger, self.state.callbacks)

        self.validate_every_n_batches = validate_every_n_batches
        self.validate_every_n_epochs = validate_every_n_epochs
        self.compute_training_metrics = compute_training_metrics
        self.grad_clip_norm = grad_clip_norm

        if deterministic_mode:
            torch.use_deterministic_algorithms(True)
            cudnn.benchmark = False
            warnings.warn("Deterministic mode is activated. This will negatively impact performance.",
                          category=UserWarning)

        # run INIT event before optimizers and schedulers are created
        self.engine.run_event(Event.INIT)

        assert isinstance(self.state.train_dataloader.dataset, collections.abc.Sized)
        steps_per_epoch = len(self.state.train_dataloader.dataset) // train_batch_size
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
            self.checkpointer = Checkpointer(checkpoint_folder=get_relative_to_run_directory(checkpoint_folder),
                                             checkpoint_interval=checkpoint_interval,
                                             checkpoint_interval_unit=checkpoint_interval_unit)

        self.checkpoint_loader = None
        if checkpoint_filepath:
            self.checkpoint_loader = CheckpointLoader(checkpoint_filepath=checkpoint_filepath)
            self.checkpoint_loader.load_checkpoint(state=self.state)

    @classmethod
    def create_from_hparams(cls, hparams: TrainerHparams) -> Trainer:
        """Instantiate a Trainer using a `TrainerHparams` object.

        Args:
            hparams (TrainerHparams): The TrainerHparams object used to instantiate the trainer.

        Returns:
            A Trainer object initialized with the provided TrainerHparams.
        """

        hparams.validate()

        # devices and systems
        device = hparams.device.initialize_object()

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

            # device
            device=device,

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

            # ddp hparams
            ddp_sync_strategy=hparams.ddp_sync_strategy,
            ddp_timeout=hparams.ddp_timeout,

            # Randomness
            seed=seed,
            deterministic_mode=hparams.deterministic_mode,

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
        original_model = self.state.model.module
        assert isinstance(original_model, BaseMosaicModel)

        metrics = original_model.metrics(train=is_train)
        assert isinstance(metrics, (Metric, MetricCollection)), \
            "Error module.metrics() must return a Metric or MetricCollection object."
        if isinstance(metrics, Metric):
            # Forcing metrics to be a MetricCollection simplifies logging results
            metrics = MetricCollection([metrics])

        # Safety check to ensure the metric and data are on the same device. Normally not
        # needed because the metric is automatically on the same device as the model.
        # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html for details.
        metrics = self.device.module_to_device(metrics)
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
        warnings.filterwarnings(action="ignore", message=r"^DataloaderMultipleIterationWarning", append=True)
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
        assert state.schedulers is not None
        # place the state, model in the proper devices
        state.model = self.device.module_to_device(state.model)
        state.optimizers = map_collection(state.optimizers, self.device.optimizer_to_device)

        # wrap model with DDP
        state.model = ddp.prepare_module(state.model, self.find_unused_parameters)
        original_model = state.model.module
        assert isinstance(original_model, BaseMosaicModel)

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

        if self._use_closures():

            def _ddp_reduce_scalar_and(flag: bool) -> bool:
                value = 1 if flag else 0
                flag_tensor = self.device.tensor_to_device(torch.tensor(value).int())
                ddp.all_reduce(flag_tensor, reduce_operation='PRODUCT')
                return flag_tensor.item() == 1

            def _ddp_reduce_tensor_sum(tensor: Tensor) -> Tensor:
                # Happens in-place; that's fine
                ddp.all_reduce(tensor, reduce_operation="SUM")
                return tensor

            state.scaler = ClosureGradScaler(ddp_reduce_scalar_and=_ddp_reduce_scalar_and,
                                             ddp_reduce_tensor_sum=_ddp_reduce_tensor_sum)
        else:
            state.scaler = GradScaler()
        use_grad_scaling = self._use_grad_scaling(state.precision, state.scaler)

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
                        state.model.eval()
                        with torch.no_grad():
                            eval_microbatches = self.train_dl_spec.split_fn(state.batch, state.grad_accum)
                            for eval_microbatch in eval_microbatches:
                                # TODO: Detect if self.run_event(Event.AFTER_DATALOADER) changes the training
                                # data and if so print a warning that metrics may return unexpected results
                                outputs, targets = original_model.validate(eval_microbatch)
                                train_metrics.update(outputs, targets)

                    state.model.train()

                    self.engine.run_event(Event.AFTER_DATALOADER)

                    microbatches = self.train_dl_spec.split_fn(state.batch, state.grad_accum)

                    self.engine.run_event(Event.BATCH_START)
                    self.logger.metric_batch({
                        "trainer/global_step": self.state.step,
                        "trainer/batch_idx": self.state.batch_idx,
                    })
                    total_loss = None
                    if self._use_closures():
                        closure = lambda **kwargs: self._train_batch(microbatches, **kwargs)
                        for optimizer in ensure_tuple(state.optimizers):
                            if use_grad_scaling:
                                total_loss = state.scaler.step(optimizer, closure=closure)
                            else:
                                # Torch optimizers technically expect closures to return a float, not a Tensor.
                                # In practice, this doesn't seem to actually matter.
                                total_loss = optimizer.step(closure=closure)  # type: ignore
                    else:
                        total_loss = self._train_batch(microbatches)
                        for optimizer in ensure_tuple(state.optimizers):
                            if use_grad_scaling:
                                state.scaler.step(optimizer)
                            else:
                                optimizer.step()

                    if use_grad_scaling:
                        state.scaler.update()

                    if total_loss is not None:
                        assert isinstance(total_loss, Tensor)

                        # total_loss can be None if gradient scaling failed
                        ddp.all_reduce(total_loss, reduce_operation="SUM")
                        ddp.barrier()
                        full_loss = total_loss.cpu().item()
                        self.logger.metric_batch({'loss/train': full_loss / ddp.get_world_size()})

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
                                                          config=self.config)
            except BreakEpochException:
                log.info(f'Skipping the rest of Epoch {state.epoch}')

            state.schedulers.step(interval='epoch')  # type: ignore
            self.engine.run_event(Event.EPOCH_END)

            if self.validate_every_n_epochs > 0 and (state.epoch + 1) % self.validate_every_n_epochs == 0:
                self.eval(is_batch=False)

            state.epoch += 1

            if self.checkpointer and self.checkpointer.should_checkpoint(state=state, event=Event.EPOCH_END):
                self.checkpointer.save_checkpoint(state=state, seed=self.seed, device=self.device, config=self.config)

        self.engine.run_event(Event.TRAINING_END)

    def _train_batch(self, microbatches: Sequence[Batch], ddp_sync: bool = True):
        """Run training on a full batch of data.

        Args:
            microbatches (Sequence[Batch]): The microbatches which make up the batch.
            ddp_sync (bool): True to sync gradients between devices on every backwards
                pass and False to only sync gradients after each device has finished
                computing a gradient on it's entire set of microbatches. (default: ``True``)
        """
        if ddp_sync or not isinstance(self.state.model, DistributedDataParallel):
            context = contextlib.nullcontext
        else:
            context = self.state.model.no_sync

        with context():  # type: ignore - Pyright apparently doesn't recognize no_sync
            return self._train_batch_inner(microbatches)

    def _train_batch_inner(self, microbatches: Sequence[Batch]):
        """Iterate over microbatches and compute the loss that will be used to step
        the optimizer.

        Args:
            microbatches (Sequence[Batch]): The microbatches which make up the batch.
        """
        self.engine.run_event(Event.BEFORE_TRAIN_BATCH)

        state = self.state
        original_model = state.model.module
        assert isinstance(original_model, BaseMosaicModel)
        assert state.optimizers is not None
        assert state.scaler is not None

        use_grad_scaling = self._use_grad_scaling(state.precision, state.scaler)

        for optimizer in ensure_tuple(state.optimizers):
            optimizer.zero_grad()

        # tracker for gradient accumulation
        total_loss = self.device.tensor_to_device(torch.zeros(size=(1,)))
        current_batch_size = sum([self._get_batch_size(batch) for batch in microbatches])

        for microbatch_idx, state.batch in enumerate(microbatches):
            is_final_microbatch = microbatch_idx + 1 == len(microbatches)
            with ddp.sync_context(state, is_final_microbatch, self.ddp_sync_strategy):
                last_microbatch_size = self._get_batch_size(state.batch)

                # forward pass
                self.engine.run_event(Event.BEFORE_FORWARD)

                with state.precision_context(state.precision):
                    state.outputs = state.model.forward(state.batch)

                self.engine.run_event(Event.AFTER_FORWARD)

                # loss
                self.engine.run_event(Event.BEFORE_LOSS)

                with state.precision_context(state.precision):
                    state.loss = original_model.loss(state.outputs, state.batch)

                for loss in ensure_tuple(state.loss):
                    loss.mul_(last_microbatch_size / current_batch_size)

                # Loss is added to losses with clone to not scale the loss for the step printout
                # Likely need to look into the performance impact
                for loss in ensure_tuple(state.loss):
                    total_loss += loss.detach().clone()

                assert state.loss is not None
                self.engine.run_event(Event.AFTER_LOSS)

                # backward
                self.engine.run_event(Event.BEFORE_BACKWARD)

                if use_grad_scaling:
                    state.loss = state.scaler.scale(state.loss)

                for loss in ensure_tuple(state.loss):
                    loss.backward(create_graph=self.backwards_create_graph)

                self.engine.run_event(Event.AFTER_BACKWARD)

        # Unscale gradients before `Event.AFTER_TRAIN_BATCH`
        if use_grad_scaling:
            for optimizer in ensure_tuple(state.optimizers):
                state.scaler.unscale_(optimizer)

        # clip gradients if the magnitude is too large
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=state.model.parameters(),
                max_norm=self.grad_clip_norm,
            )

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

        model.eval()
        with torch.no_grad():

            self.engine.run_event(Event.EVAL_START)

            original_model = state.model.module
            assert isinstance(original_model, BaseMosaicModel)

            metrics = self._get_metrics_as_collection(is_train=False)

            assert state.eval_dataloader is not None

            for state.batch in state.eval_dataloader:
                self.engine.run_event(Event.EVAL_BATCH_START)

                self.engine.run_event(Event.EVAL_BEFORE_FORWARD)
                state.outputs, targets = original_model.validate(state.batch)
                self.engine.run_event(Event.EVAL_AFTER_FORWARD)

                metrics.update(state.outputs, targets)

                self.engine.run_event(Event.EVAL_BATCH_END)

            self._compute_and_log_metrics(metrics, is_train=False, is_batch=is_batch)
            self.engine.run_event(Event.EVAL_END)

        if restore_model_train:
            model.train()

    def _use_grad_scaling(self, precision: Union[str, Precision], scaler: Optional[GradScaler]) -> bool:
        """Determines based on precision when to use grad scaling.

        By default, the pytorch GradScaler is a no-op if running on
        unsupported hardware. Here we raise a RuntimeError instead.

        Args:
            precision (Precision): Numerical precision, based on the Precision Enum.
            scaler (GradScaler): Used to make sure that the scaler is enabled when
            using grad scaling.

        Raises:
            RuntimeError:
                Occurs when attempting to use grad scaling without the scaler
                enabled. Likely due to hardware not supporting the provided precision.
        """
        precision = Precision(precision)
        use_grad_scaling = precision == Precision.AMP

        if use_grad_scaling and (scaler is None or not scaler.is_enabled()):
            raise RuntimeError(f'Attempting to use grad scaling with {precision}, but scaler is not enabled.'
                               f'Potentially your hardware does not support Precision {precision}.')
        return use_grad_scaling

    def _use_closures(self) -> bool:
        """Determines based on precision and optimizers whether to use closures.

        We default to using closures unless AMP is enabled, in which case we only allow
        closures when using optimizers with the _step_supports_amp_closure flag.
        """
        if self.state.precision != Precision.AMP:
            return True

        if self.state.optimizers is None:
            raise RuntimeError("state.optimizers must be set before `_use_closures` can be determined")

        return all(
            getattr(optimizer, "_step_supports_amp_closure", False)
            for optimizer in ensure_tuple(self.state.optimizers))
