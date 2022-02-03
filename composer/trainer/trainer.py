# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import contextlib
import datetime
import itertools
import logging
import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Optional, Sequence, Union, cast

import torch
import torch.distributed
import torch.utils.data
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric

from composer.core import Callback, DataSpec, Engine, Event, Logger, State, Time, surgery
from composer.core.algorithm import Algorithm
from composer.core.evaluator import Evaluator
from composer.core.logging import BaseLoggerBackend, LogLevel
from composer.core.time import TimeUnit
from composer.core.types import (Batch, BreakEpochException, DataLoader, Evaluators, Metrics, Optimizers, Precision,
                                 Schedulers)
from composer.datasets.dataloader import unwrap_data_loader
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.models.base import ComposerModel
from composer.optim import ComposedScheduler
from composer.optim.decoupled_weight_decay import DecoupledSGDW
from composer.profiler.profiler_hparams import ProfilerHparams
from composer.trainer.checkpoint import CheckpointLoader, CheckpointSaver
from composer.trainer.ddp import DDPSyncStrategy, ddp_sync_context, prepare_ddp_module
from composer.trainer.deepspeed import fix_batch_precision_for_deepspeed, parse_deepspeed_config
from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.scaler import ClosureGradScaler
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist, ensure_tuple, map_collection, reproducibility
from composer.utils.object_store import ObjectStoreProvider

if TYPE_CHECKING:
    import deepspeed

log = logging.getLogger(__name__)


class Trainer:
    """Trainer for training a model with algorithms.

    Can be created either with ``__init__`` or by providing a
    :class:`~composer.trainer.TrainerHparams` object
    (see :meth:`~composer.trainer.Trainer.create_from_hparams`).

    Args:
        model (ComposerModel): The model to train.
        train_dataloader (DataLoader, DataSpec, or dict): The :class:`DataLoader`, :class:`DataSpec`,
            or dict of :class:`DataSpec` kwargs for the training data.
        eval_dataloader (DataLoader, DataSpec, Evaluators): The :class:`DataLoader`, :class:`DataSpec`,
            :class:`Evaluators` for the evaluation data. The :class:`Evaluator`
            class contains metrics relevant to the specific dataset. Set to ``None`` for no evaluation.
        max_duration (Time or str): The maximum duration to train. See `~composer.core.Time` for details.
        algorithms (List[Algorithm], optional): The algorithms to use during training.
            (default: ``[]``)
        optimizers: (Optimizers, optional): The optimizers.
            (default: ``DecoupledSGDW(model.parameters(), lr=0.1)``)
        schedulers: (Schedulers, optional): The schedulers.
            (default: ``[CosineAnnealingLR()]``).
        device (str or Device, optional): The device to use for training. Either `cpu` or `gpu`.
            (default `cpu`)
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
        precision (str or Precision, optional): Numerical precision to use for training, one of 'fp32', 'fp16'
            for 'amp' (recommended). (default: ``Precision.FP32``).
        dist_timeout (float, optional): Timeout, in seconds, for initializing the distributed process group.
            (default: ``15.0``)
        ddp_sync_strategy (str or DDPSyncStrategy, optional): The strategy to use for synchronizing gradients.
            Leave unset to let the trainer auto-configure this. For more details, see ``DDPSyncStrategy``.
        seed (int, optional): The seed used in randomization. When not provided a random seed
            will be created. (default: ``None``)
        deterministic_mode (bool, optional): Run the model deterministically. Experimental. Performance
            degradations expected. Certain Torch modules may not have deterministic implementations,
            which will result in a crash. (default: ``False``)
        log_destinations (List[BaseLoggerBackend], optional): The destinations to log training information to.
            (default: ``[TQDMLoggerBackend()]``).
        callbacks (Sequence[Callback], optional): The callbacks to run during training. (default: ``[]``)
        load_path (str, optional): Path to a specific checkpoint to load. If not set (the default),
            then no checkpoint will be loaded. (default: ``None``)
        load_object_store (ObjectStoreProvider, optional): For loading from object stores (e.g. S3), this
            ObjectStoreProvider instance that will be used to download the checkpoint. Ignored if
            ``load_path`` is not specified. (default: ``None``)
        load_weights_only (bool): Only load the model weights.  Ignored if ``load_path`` is not specified.
            (default: ``False``)
        load_strict (bool): Ensure that the set of weights in the checkpoint and model must exactly match. Ignored if
            ``load_path`` is not specified. (default: ``False``)
        load_chunk_size (int): Chunk size (in bytes) to use when downloading checkpoints.
            Ignored if the ``load_path`` is not specified or it is a local file path. (default: ``1,048,675``)
        load_progress_bar (bool): Display the progress bar for downloading the checkpoint. Ignored if
            ``load_path`` is not specified or if it is a local file path. (default: ``True``)
        save_folder (str, optional): Folder path to save checkpoints, relative to the run directory.
            Set to ``None`` to not save checkpoints. (default: ``None``)
        save_interval (str): How often to save checkpoints. For example, set to "1ep" to save checkpoints
            every epoch, or "10ba" to save checkpoints every 10 batches. (default: ``1ep``)
        save_interval_unit (str): Unit of ``save_interval``. Can be ``ep`` or ``steps``. (default: ``ep``).
        save_compression (str): Compression algorithm to run on checkpoints. Can be `gzip`, `bzip2`,
            `lzma`, or left blank for no compression.  (default: ``""`` for no compression).
        train_subset_num_batches (int, optional): If specified, finish every epoch early after training
            on this many batches. This parameter has no effect if it is greater than ``len(train_dataloader)``.
            If None (the default), then the entire dataloader will be iterated over.
        eval_subset_num_batches (int, optional): If specified, evaluate on this many batches.
            This parameter has no effect if it is greater than ``len(eval_dataloader)``.
            If None (the default), then the entire dataloader will be iterated over.
        deepspeed_config (Dict[str, Any], optional): Configuration for DeepSpeed, formatted as a JSON
            according to `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_. If any
            non-None value is provided, the trainer will initialize the DeepSpeed engine. (default: ``None``)
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
            model: ComposerModel,
            train_dataloader: Union[DataLoader, DataSpec],
            eval_dataloader: Optional[Union[DataLoader, DataSpec, Evaluators]],
            max_duration: Union[str, Time],
            algorithms: Optional[List[Algorithm]] = None,
            optimizers: Optional[Optimizers] = None,
            schedulers: Optional[Schedulers] = None,

            # device
            device: Optional[Union[str, Device]] = None,

            # training hparams
            grad_accum: int = 1,
            grad_clip_norm: Optional[float] = None,
            validate_every_n_batches: int = -1,
            validate_every_n_epochs: int = 1,
            compute_training_metrics: bool = False,
            precision: Union[str, Precision] = Precision.FP32,

            # dist hparams
            dist_timeout: float = 300.0,
            ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]] = None,

            # Randomness
            seed: Optional[int] = None,
            deterministic_mode: bool = False,

            # Logging and callbacks
            log_destinations: Optional[Sequence[BaseLoggerBackend]] = None,
            callbacks: Sequence[Callback] = tuple(),

            # load checkpoint
            load_path: Optional[str] = None,
            load_object_store: Optional[ObjectStoreProvider] = None,
            load_weights_only: bool = False,
            load_strict: bool = False,
            load_chunk_size: int = 1_048_576,
            load_progress_bar: bool = True,

            # save_checkpoint
            save_folder: Optional[str] = None,
            save_interval: str = "1ep",
            save_compression: str = '',

            # Profiling
            profiler: Optional[ProfilerHparams] = None,

            # Subset parameters
            train_subset_num_batches: Optional[int] = None,
            eval_subset_num_batches: Optional[int] = None,

            # DeepSpeed
            deepspeed_config: Optional[Dict[str, Any]] = None,

            # Optional config (ex. an hparams yaml file)
            config: Optional[Dict[str, Any]] = None):
        # surpressing GradScaler warnings as they are always created
        # self._use_grad_scaling() will raise a RuntimeError if grad scaling is not available when it is required
        warnings.filterwarnings(action="ignore", message="torch.cuda.amp.GradScaler")

        if isinstance(max_duration, str):
            max_duration = Time.from_timestring(max_duration)

        self.config = config

        self.deepspeed_config = deepspeed_config

        if not device:
            self.device = DeviceCPU() if not self.deepspeed_enabled else DeviceGPU()
        elif isinstance(device, str):
            if device == 'cpu':
                self.device = DeviceCPU()
            elif device == 'gpu':
                self.device = DeviceGPU()
            else:
                raise ValueError(f'device ({device}) must be one of (cpu, gpu).')
        else:
            if not isinstance(device, Device):
                raise ValueError('device must be of class Device')
            self.device = device

        if not seed:
            seed = reproducibility.get_random_seed()
            log.info(f"Seed was None. Setting seed to random value: {seed}")

        # Assure that each process has a different seed, necessary if a seed is passed to init
        seed += dist.get_global_rank()

        # If hparams is used to create the Trainer this function is called twice
        # which is okay because all runs with the hparams codepath will do this
        reproducibility.seed_all(seed)
        self.seed = seed

        if not algorithms:
            algorithms = []

        self.backwards_create_graph = any(map(lambda x: x.backwards_create_graph, algorithms))

        find_unused_parameters = any(map(lambda x: x.find_unused_parameters, algorithms))

        self.find_unused_parameters = find_unused_parameters

        if self.deepspeed_enabled:
            import deepspeed
            deepspeed.init_distributed()
        else:
            dist.initialize_dist(self.device.dist_backend, datetime.timedelta(seconds=dist_timeout))
            if ddp_sync_strategy is None:
                self.ddp_sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC if not find_unused_parameters else DDPSyncStrategy.FORCED_SYNC
            else:
                self.ddp_sync_strategy = DDPSyncStrategy(ddp_sync_strategy)

        # `eval_dataloader` could be a dataloader, dataspec, evaluator, List[Evaluator], Tuple[Evaluator, ...], or dict of Dataspec hparams
        # convert it to `List[Evaluator]`
        self.evaluators: List[Evaluator] = []
        for evaluator in ensure_tuple(eval_dataloader):
            if isinstance(evaluator, Evaluator):
                self.evaluators.append(evaluator)
            else:
                metrics = model.metrics(train=False)
                default_evaluator = Evaluator(label="eval_dataset", dataloader=evaluator, metrics=metrics)
                self.evaluators.append(default_evaluator)

        # do a check here to make sure there is at least one validation set
        if len(self.evaluators) == 0:
            warnings.warn(
                textwrap.dedent("""No evaluation dataset was specified. Please specify `eval_dataloader` to periodically
                evaluate your model while training."""),
                category=UserWarning)

        # TODO(#123): DeepSpeed still needs a precision context, but it's not completely clear how to
        # handle this with our version of Pytorch
        precision_context = self.device.precision_context if not self.deepspeed_enabled else cast(
            Callable[..., ContextManager], contextlib.nullcontext)
        if isinstance(precision, str):
            precision = Precision(precision)

        if not isinstance(train_dataloader, DataSpec):
            train_dataloader = DataSpec(train_dataloader)

        self._train_data_spec = train_dataloader
        unwrapped_data_loader = unwrap_data_loader(self._train_data_spec.dataloader)
        if isinstance(unwrapped_data_loader, torch.utils.data.DataLoader):
            if unwrapped_data_loader._iterator is not None:
                raise ValueError(
                    textwrap.dedent("""\
                    The `train_dataloader` has an active iterator. This could occur
                    if `persistent_workers=True` and the dataloader has already been iterated,
                    or if the dataloader is mid-epoch. It is required that the training dataloader
                    does not have an active iterator, so CPU dataset augmentations can be
                    correctly inserted.

                    To fix, please do not iterate over the dataloader before passing it into
                    the trainer."""))

        if eval_subset_num_batches is not None:
            for evaluator in self.evaluators:
                try:
                    eval_dataloader_len = len(evaluator.dataloader.dataloader)
                except (NotImplementedError, TypeError):
                    pass
                else:
                    if eval_subset_num_batches > eval_dataloader_len:
                        warnings.warn(
                            textwrap.dedent(
                                f"""SubsetNumBatchesWarning: The eval_subset_num_batches({eval_subset_num_batches})
                                is greater than the number of batches in the evaluator ({evaluator.label}) dataloader
                                ({len(evaluator.dataloader.dataloader)})"""))
        self._eval_subset_num_batches = eval_subset_num_batches

        if not optimizers:
            optimizers = DecoupledSGDW(list(model.parameters()), lr=0.1)
            warnings.warn(f"No optimizer was specified. Defaulting to {repr(optimizers)}")

        num_optimizers = len(ensure_tuple(optimizers))

        if num_optimizers != 1:
            raise NotImplementedError(f"Only one optimizer is supported; found {num_optimizers} optimizers")

        if not schedulers:
            optimizer = ensure_tuple(optimizers)[0]
            if not max_duration.unit == TimeUnit.EPOCH:
                raise ValueError("If a scheduler is not provided, max duration must be in epochs")
            schedulers = CosineAnnealingLR(optimizer, T_max=max_duration.value)
            warnings.warn(f"No scheduler was specified. Defaulting to {repr(schedulers)}")
        if not isinstance(schedulers, (tuple, list)):
            schedulers = [schedulers]
        schedulers = ComposedScheduler(schedulers)

        self.state = State(
            max_duration=max_duration,
            algorithms=algorithms,
            model=model,
            callbacks=callbacks,
            grad_accum=grad_accum,
            precision=precision,
            precision_context=precision_context,
            train_dataloader=train_dataloader.dataloader,
            evaluators=self.evaluators,
            optimizers=optimizers,
            steps_per_epoch=train_subset_num_batches,
            schedulers=schedulers,
        )

        # Configure the profiler
        if profiler is not None:
            self.state.profiler = profiler.initialize_object(self.state)
            self.state.callbacks.extend(self.state.profiler.event_handlers)

        if log_destinations is None:
            log_destinations = [TQDMLoggerBackend()]
        self.logger = Logger(self.state, log_destinations)
        self.state.callbacks = list(cast(List[Callback], log_destinations)) + self.state.callbacks

        self.engine = Engine(
            state=self.state,
            logger=self.logger,
        )

        self.validate_every_n_batches = validate_every_n_batches
        self.validate_every_n_epochs = validate_every_n_epochs
        self.compute_training_metrics = compute_training_metrics
        self.grad_clip_norm = grad_clip_norm

        if deterministic_mode:
            reproducibility.configure_deterministic_mode()

        self.engine.run_event(Event.INIT)

        assert isinstance(self.state.model, ComposerModel)
        self.original_model = self.state.model  # TODO(ravi) -- update the state to add an original model helper

        self.checkpoint_saver = None
        if save_folder is not None:
            self.checkpoint_saver = CheckpointSaver(
                save_folder=save_folder,
                interval=save_interval,
                compression=save_compression,
            )

        self.checkpoint_loader = None
        if load_path is not None:
            self.checkpoint_loader = CheckpointLoader(path=load_path,
                                                      object_store=load_object_store,
                                                      load_weights_only=load_weights_only,
                                                      strict_model_weights=load_strict,
                                                      chunk_size=load_chunk_size,
                                                      progress_bar=load_progress_bar)

        # place the state, model in the proper devices, and initialize from a checkpoint if provided
        if self.deepspeed_enabled:
            import deepspeed
            assert deepspeed_config is not None
            self.deepspeed_config = parse_deepspeed_config(deepspeed_config,
                                                           state=self.state,
                                                           grad_clip_norm=self.grad_clip_norm)
            optimizer = ensure_tuple(self.state.optimizers)[0]
            (self.state.model, self.state.optimizers, _, _) = deepspeed.initialize(
                config=self.deepspeed_config,
                model=self.state.model,
                optimizer=optimizer,
            )

        # If using DeepSpeed, the model must be loaded from checkpoint after the engine has been
        # initialized, but if using PyTorch DDP, the model must be loaded before it is wrapped with
        # DDP.
        if self.checkpoint_loader:
            restored_seed = self.checkpoint_loader.load_checkpoint(state=self.state)
            if restored_seed is not None:
                self.seed = restored_seed

        if not self.deepspeed_enabled:
            host_model_params = self.state.model.parameters()
            self.state.model = self.device.module_to_device(self.state.model)
            device_model_params = self.state.model.parameters()

            # use surgery to update the parameters of the optimizers, now that the model is on the device
            # see https://pytorch.org/docs/stable/optim.html#constructing-it
            surgery.replace_params_in_optimizer(old_params=host_model_params,
                                                new_params=device_model_params,
                                                optimizers=self.state.optimizers)

            # Move any remaining optimizer parameters onto the device
            self.state.optimizers = map_collection(self.state.optimizers, self.device.optimizer_to_device)

            # wrap model with DDP
            self.state.model = prepare_ddp_module(self.state.model, self.find_unused_parameters)
        

    @property
    def deepspeed_enabled(self):
        return self.deepspeed_config is not None

    def fit(self):
        """Train and evaluate the model on the provided data."""
        try:
            self._train_loop()
        finally:
            self.engine.close()

    def _ensure_metrics_device_and_dtype(self, metrics: MetricCollection):
        # Safety check to ensure the metric and data are on the same device. Normally not
        # needed because the metric is automatically on the same device as the model.
        # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html for details.
        metrics = self.device.module_to_device(metrics)

        # HACK: DeepSpeed somehow manages to convert metric internal states to its own dtype. When
        # running with FP16, this tends to result in overflows. Let's assume FP32 is good enough.
        for _, metric in metrics.items():
            metric.set_dtype(torch.float32)  # type: ignore

        return metrics

    def _compute_and_log_metrics(self, metrics: Metrics, *, is_train: bool, is_batch: bool, logging_label: str = ''):
        """Computes metrics, logs the results, and resets the metrics.

        Args:
            metrics (Metrics): The metrics to compute.
            is_train (bool): True for training metrics, False for evaluation metrics.
            is_batch (bool): True if logging at batch level, false for epoch level.
            logging_label (str): Should be left as empty string if called for training metrics.
                Should be the evaluator label if called on evaluator metrics.
        """
        computed_metrics = metrics.compute()
        for name, value in computed_metrics.items():
            log_level = LogLevel.BATCH if is_batch else LogLevel.EPOCH
            suffix = 'train' if is_train else 'val'

            # default label given to evaluator created by val_dataset parameter
            if not logging_label or logging_label == "eval_dataset":
                label = f'{name.lower()}/{suffix}'
            else:
                label = f'{logging_label}/{name.lower()}_{suffix}'
            self.logger.metric(log_level, {label: value})
        metrics.reset()

    def _spin_dataloaders(self):
        """Spin the dataloaders to restore sampler state.

        Only one batch must be loaded to seed the sampler's generator. since only the first batch is being loaded, the
        dataloader may not be completely iterated through.
        """
        # spin the evaluator dataloaders once to initialize its sampler deterministically
        # so it does not affect any other RNG reads
        for evaluator in self.state.evaluators:
            dataloader = evaluator.dataloader.dataloader
            if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                dataloader.sampler.set_epoch(0)
            for _ in dataloader:
                break

        # spin the train dataloader's sampler to get to the state of the desired epoch
        for epoch in range(int(self.state.timer.epoch)):
            if isinstance(self.state.train_dataloader.sampler, torch.utils.data.DistributedSampler):
                self.state.train_dataloader.sampler.set_epoch(epoch)
            for _ in self.state.train_dataloader:
                break

    def _train_loop(self) -> None:
        """Run training for the specified number of epochs and log results."""
        # shorthand
        state = self.state

        # print training start
        self.logger.metric_fit({"trainer/algorithms": [str(algo) for algo in self.state.algorithms]})

        if self.compute_training_metrics:
            log.warn('Computing model evaluation metrics during training.'
                     ' This doubles the number of forward passes and may lead'
                     ' to a throughput degradation.')
            train_metrics = self.original_model.metrics(train=False)
            if isinstance(train_metrics, Metric):
                # Forcing metrics to be a MetricCollection simplifies logging results
                train_metrics = MetricCollection([train_metrics])

            train_metrics = self._ensure_metrics_device_and_dtype(train_metrics)
        else:
            train_metrics = None

        self.engine.run_event(Event.TRAINING_START)

        state.scaler = ClosureGradScaler() if self._use_closures() else GradScaler()
        use_grad_scaling = self._use_grad_scaling(state.precision, state.scaler)

        self._spin_dataloaders()

        if self.state.timer.batch_in_epoch == 0 and self.checkpoint_loader:
            # only restore the rng state here if the step in the current epoch is zero.
            self.checkpoint_loader.restore_checkpoint_rng_state(self.device)

        while state.timer < state.max_duration:
            try:
                state.model.train()

                if self.state.timer.batch_in_epoch == 0:
                    self.engine.run_event(Event.EPOCH_START)
                    self.logger.metric_epoch({"epoch": self.state.epoch})

                if isinstance(self.state.train_dataloader.sampler, torch.utils.data.DistributedSampler):
                    self.state.train_dataloader.sampler.set_epoch(int(self.state.timer.epoch))

                for batch_idx, state.batch in enumerate(
                        itertools.islice(state.train_dataloader, self.state.steps_per_epoch)):

                    # if resuming, skip dataloader forward to the minibatch index
                    if batch_idx < self.state.timer.batch_in_epoch:
                        if self.checkpoint_loader:
                            self.checkpoint_loader.restore_checkpoint_rng_state(self.device)
                        continue

                    state.batch = self.device.batch_to_device(state.batch)
                    state.batch = self._train_data_spec.device_transforms(state.batch)
                    state.batch_num_samples = self._train_data_spec.get_num_samples_in_batch(state.batch)
                    state.batch_num_tokens = self._train_data_spec.get_num_tokens_in_batch(state.batch)

                    if self.deepspeed_enabled:
                        state.batch = fix_batch_precision_for_deepspeed(state.batch, state.precision)

                    if self.compute_training_metrics:
                        # compute metrics on the training set
                        assert train_metrics is not None
                        state.model.eval()
                        with torch.no_grad():
                            for eval_microbatch in self._train_data_spec.split_batch(state.batch, state.grad_accum):
                                # TODO: Detect if self.run_event(Event.AFTER_DATALOADER) changes the training
                                # data and if so print a warning that metrics may return unexpected results
                                outputs, targets = self.original_model.validate(eval_microbatch)
                                train_metrics.update(outputs, targets)

                    state.model.train()

                    self.engine.run_event(Event.AFTER_DATALOADER)

                    num_samples_in_batch = self.device.tensor_to_device(
                        torch.tensor([state.batch_num_samples], dtype=torch.int))
                    num_tokens_in_batch = self.device.tensor_to_device(
                        torch.tensor([state.batch_num_tokens], dtype=torch.int))
                    dist.all_reduce(num_samples_in_batch, reduce_operation="SUM")
                    dist.all_reduce(num_tokens_in_batch, reduce_operation="SUM")

                    self.engine.run_event(Event.BATCH_START)
                    self.logger.metric_batch({
                        "trainer/global_step": self.state.step,
                        "trainer/batch_idx": self.state.timer.batch_in_epoch.value,
                    })
                    total_loss = None
                    microbatches = self._train_data_spec.split_batch(state.batch, state.grad_accum)
                    if self.deepspeed_enabled:
                        total_loss = self._train_batch(microbatches)
                    elif self._use_closures():
                        for optimizer in state.optimizers:
                            if use_grad_scaling:
                                total_loss = state.scaler.step(
                                    optimizer, closure=lambda **kwargs: self._train_batch(microbatches, **kwargs))
                            else:
                                total_loss = optimizer.step(
                                    closure=lambda **kwargs: self._train_batch(microbatches, **kwargs).item())
                    else:
                        total_loss = self._train_batch(microbatches)
                        for optimizer in state.optimizers:
                            if use_grad_scaling:
                                state.scaler.step(optimizer)
                            else:
                                optimizer.step()

                    if use_grad_scaling:
                        state.scaler.update()

                    if total_loss is not None:
                        if not isinstance(total_loss, torch.Tensor):
                            total_loss = self.device.tensor_to_device(torch.tensor([total_loss]))

                        # total_loss can be None if gradient scaling failed
                        dist.all_reduce(total_loss, reduce_operation="SUM")
                        full_loss = total_loss.cpu().item()
                        self.logger.metric_batch({'loss/train': full_loss / dist.get_world_size()})

                    if self.compute_training_metrics:
                        assert train_metrics is not None
                        self._compute_and_log_metrics(train_metrics, is_train=True, is_batch=True)

                    state.timer.on_batch_complete(
                        samples=int(num_samples_in_batch.item()),
                        tokens=int(num_tokens_in_batch.item()),
                    )

                    for scheduler in state.schedulers:
                        scheduler.step(interval='batch')  # type: ignore

                    self.engine.run_event(Event.BATCH_END)

                    if self.validate_every_n_batches > 0 and int(
                            state.timer.batch) % self.validate_every_n_batches == 0:
                        self.eval(is_batch=True)

                    if self.checkpoint_saver and self.checkpoint_saver.should_checkpoint(state=state,
                                                                                         event=Event.BATCH_END):
                        self.checkpoint_saver.save_checkpoint(state=state,
                                                              seed=self.seed,
                                                              device=self.device,
                                                              config=self.config)
            except BreakEpochException:
                log.info(f'Skipping the rest of Epoch {state.epoch}')

            state.timer.on_epoch_complete()

            for scheduler in state.schedulers:
                scheduler.step(interval='epoch')  # type: ignore

            self.engine.run_event(Event.EPOCH_END)

            if self.validate_every_n_epochs > 0 and int(state.timer.epoch) % self.validate_every_n_epochs == 0:
                self.eval(is_batch=False)

            if self.checkpoint_saver and self.checkpoint_saver.should_checkpoint(state=state, event=Event.EPOCH_END):
                self.checkpoint_saver.save_checkpoint(state=state,
                                                      seed=self.seed,
                                                      device=self.device,
                                                      config=self.config)

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
            context = cast(Callable[[], ContextManager], self.state.model.no_sync)

        with context():
            return self._train_batch_inner(microbatches)

    def _train_batch_inner(self, microbatches: Sequence[Batch]):
        """Iterate over microbatches and compute the loss that will be used to step the optimizer."""
        self.engine.run_event(Event.BEFORE_TRAIN_BATCH)

        state = self.state
        assert state.optimizers is not None
        assert state.scaler is not None

        use_grad_scaling = self._use_grad_scaling(state.precision, state.scaler)

        if not self.deepspeed_enabled:
            for optimizer in state.optimizers:
                optimizer.zero_grad()

        # tracker for gradient accumulation
        total_loss = self.device.tensor_to_device(torch.zeros(size=(1,)))
        current_batch_size = sum([self._train_data_spec.get_num_samples_in_batch(batch) for batch in microbatches])

        for microbatch_idx, state.batch in enumerate(microbatches):
            state.batch_num_tokens = self._train_data_spec.get_num_tokens_in_batch(state.batch)
            state.batch_num_samples = self._train_data_spec.get_num_samples_in_batch(state.batch)
            is_final_microbatch = microbatch_idx + 1 == len(microbatches)
            sync_context = contextlib.nullcontext() if self.deepspeed_enabled else ddp_sync_context(
                state, is_final_microbatch, self.ddp_sync_strategy)
            with sync_context:
                # forward pass
                self.engine.run_event(Event.BEFORE_FORWARD)

                with state.precision_context:
                    state.outputs = state.model.forward(state.batch)

                self.engine.run_event(Event.AFTER_FORWARD)

                # loss
                self.engine.run_event(Event.BEFORE_LOSS)

                with state.precision_context:
                    state.loss = self.original_model.loss(state.outputs, state.batch)

                # We always want to scale loss by the grad_accum before the backwards pass and
                # also for sake of metrics. Complicating matters, the DeepSpeed engine does its
                # own scaling when we call `.backward`, but this isn't in place so we still need
                # to scale for sake of metrics after the `.backward` call.

                # Loss is added to losses with clone to not scale the loss for the step printout
                # Likely need to look into the performance impact
                if not self.deepspeed_enabled:
                    for loss in ensure_tuple(state.loss):
                        loss.mul_(state.batch_num_samples / current_batch_size)
                        total_loss += loss.detach().clone()

                assert state.loss is not None
                self.engine.run_event(Event.AFTER_LOSS)

                # backward
                self.engine.run_event(Event.BEFORE_BACKWARD)

                if use_grad_scaling:
                    state.loss = state.scaler.scale(state.loss)

                if self.deepspeed_enabled:
                    cast("deepspeed.DeepSpeedEngine", state.model).backward(state.loss)

                    # This is the same loss scaling and reporting we skipped earlier.
                    for loss in ensure_tuple(state.loss):
                        loss.mul_(state.batch_num_samples / current_batch_size)
                        total_loss += loss.detach().clone()
                else:
                    for loss in ensure_tuple(state.loss):
                        loss.backward(create_graph=self.backwards_create_graph)

                self.engine.run_event(Event.AFTER_BACKWARD)

            if self.deepspeed_enabled:
                cast("deepspeed.DeepSpeedEngine", state.model).step()

        # Unscale gradients before `Event.AFTER_TRAIN_BATCH`
        if use_grad_scaling:
            for optimizer in ensure_tuple(state.optimizers):
                state.scaler.unscale_(optimizer)

        # clip gradients if the magnitude is too large
        if not self.deepspeed_enabled and self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=state.model.parameters(),
                max_norm=self.grad_clip_norm,
            )

        self.engine.run_event(Event.AFTER_TRAIN_BATCH)

        return total_loss

    def eval(self, is_batch: bool):
        """Evaluate the model on the provided evaluation data and log appropriate metrics.

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

            for evaluator in state.evaluators:
                dataloader = evaluator.dataloader.dataloader
                metrics = self._ensure_metrics_device_and_dtype(evaluator.metrics)
                if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                    # The distributed sampler uses `set_epoch` to set the random seed
                    # Because evaluation can run on each batch, we use the batch to seed the sampler
                    # so each evaluation will get a proper shuffle.
                    # The epoch provided to `set_epoch` need not be sequential, so this is fine.
                    dataloader.sampler.set_epoch(int(self.state.timer.batch))

                for state.batch in itertools.islice(dataloader, self._eval_subset_num_batches):
                    state.batch = self.device.batch_to_device(state.batch)
                    if evaluator.dataloader.device_transforms:
                        state.batch = evaluator.dataloader.device_transforms(state.batch)
                    state.batch_num_samples = evaluator.dataloader.get_num_samples_in_batch(state.batch)
                    state.batch_num_tokens = evaluator.dataloader.get_num_tokens_in_batch(state.batch)

                    if self.deepspeed_enabled:
                        state.batch = fix_batch_precision_for_deepspeed(state.batch, state.precision)

                    self.engine.run_event(Event.EVAL_BATCH_START)

                    self.engine.run_event(Event.EVAL_BEFORE_FORWARD)
                    state.outputs, targets = self.original_model.validate(state.batch)
                    self.engine.run_event(Event.EVAL_AFTER_FORWARD)

                    metrics.update(state.outputs, targets)

                    self.engine.run_event(Event.EVAL_BATCH_END)

                self._compute_and_log_metrics(metrics, is_train=False, is_batch=is_batch, logging_label=evaluator.label)

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
        if self.deepspeed_enabled:
            return False

        precision = Precision(precision)
        use_grad_scaling = precision == Precision.AMP

        if use_grad_scaling and (scaler is None or not scaler.is_enabled()):
            raise RuntimeError(f'Attempting to use grad scaling with {precision}, but scaler is not enabled.'
                               f'Potentially your hardware does not support Precision {precision}.')
        return use_grad_scaling

    def _use_closures(self) -> bool:
        """Determines based on precision and optimizers whether to use closures.

        We default to using closures unless AMP is enabled, in which case we only allow closures when using optimizers
        with the _step_supports_amp_closure flag.
        """
        if self.deepspeed_enabled:
            return False

        if self.state.precision != Precision.AMP:
            return True

        if self.state.optimizers is None:
            raise RuntimeError("state.optimizers must be set before `_use_closures` can be determined")

        return all(
            getattr(optimizer, "_step_supports_amp_closure", False)
            for optimizer in ensure_tuple(self.state.optimizers))
