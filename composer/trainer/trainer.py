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
from composer.core.logging import BaseLoggerBackend, LogLevel
from composer.core.time import TimeUnit
from composer.core.types import Batch, BreakEpochException, DataLoader, Metrics, Optimizers, Precision, Schedulers
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.models.base import BaseMosaicModel
from composer.optim import ComposedScheduler
from composer.optim.decoupled_weight_decay import DecoupledSGDW
from composer.optim.scheduler import ensure_warmup_last
from composer.profiler.profiler_hparams import ProfilerHparams
from composer.trainer.checkpoint_hparams import CheckpointLoaderHparams, CheckpointSaverHparams
from composer.trainer.ddp import DDPSyncStrategy, ddp_sync_context, prepare_ddp_module
from composer.trainer.deepspeed import DeepSpeedHparams, fix_batch_precision_for_deepspeed
from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.scaler import ClosureGradScaler
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist, ensure_tuple, map_collection, reproducibility

if TYPE_CHECKING:
    import deepspeed

log = logging.getLogger(__name__)


class Trainer:
    """Trainer for training a model with algorithms.

    Can be created either with ``__init__`` or by providing a
    :class:`~composer.trainer.TrainerHparams` object
    (see :meth:`~composer.trainer.Trainer.create_from_hparams`).

    Args:
        model (BaseMosaicModel): The model to train.
        train_dataloader (DataLoader, DataSpec, or dict): The :class:`DataLoader`, :class:`DataSpec`,
            or dict of :class:`DataSpec` kwargs for the training data.
        eval_dataloader (DataLoader, DataSpec, or dict): The :class:`DataLoader`, :class:`DataSpec`,
            or dict of :class:`DataSpec` kwargs for the evaluation data.
        max_duration (Time or str): The maximum duration to train. See `~composer.core.Time` for details.
        algorithms (List[Algorithm], optional): The algorithms to use during training.
            (default: ``[]``)
        optimizers: (Optimizers, optional): The optimizers.
            (default: ``DecoupledSGDW(model.parameters(), lr=0.1)``)
        schedulers: (Schedulers, optional): The schedulers.
            (default: ``[CosineAnnealingLR()]``).
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
        dist_timeout (float, optional): Timeout, in seconds, for initializing the distributed process group.
            (default: ``15.0``)
        ddp_sync_strategy (DDPSyncStrategy, optional): The strategy to use for synchronizing gradients.
            Leave unset to let the trainer auto-configure this.
        seed (int, optional): The seed used in randomization. When not provided a random seed
            will be created. (default: ``None``)
        deterministic_mode (bool, optional): Run the model deterministically. Experimental. Performance
            degradations expected. Certain Torch modules may not have deterministic implementations,
            which will result in a crash. (default: ``False``)
        log_destinations (List[BaseLoggerBackend], optional): The destinations to log training information to.
            (default: ``[TQDMLoggerBackend()]``).
        callbacks (Sequence[Callback], optional): The callbacks to run during training. (default: ``[]``)
        checkpoint_loader (CheckpointLoaderHparams, optional): If specified, load the specified checkpoint.
            (default: ``None``)
        checkpoint_saver (CheckpointSaverHparams, optional): If specified, save checkpoints according to
            the given parameters (default: ``None``)
        train_subset_num_batches (int, optional): If specified, finish every epoch early after training
            on this many batches. This parameter has no effect if it is greater than ``len(train_dataloader)``.
            If None (the default), then the entire dataloader will be iterated over.
        eval_subset_num_batches (int, optional): If specified, evaluate on this many batches.
            This parameter has no effect if it is greater than ``len(eval_dataloader)``.
            If None (the default), then the entire dataloader will be iterated over.
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
            train_dataloader: Union[DataLoader, DataSpec],
            eval_dataloader: Union[DataLoader, DataSpec],
            max_duration: Union[str, Time],
            algorithms: Optional[List[Algorithm]] = None,
            optimizers: Optional[Optimizers] = None,
            schedulers: Optional[Schedulers] = None,

            # device
            device: Optional[Device] = None,

            # training hparams
            grad_accum: int = 1,
            grad_clip_norm: Optional[float] = None,
            validate_every_n_batches: int = -1,
            validate_every_n_epochs: int = 1,
            compute_training_metrics: bool = False,
            precision: Precision = Precision.FP32,

            # dist hparams
            dist_timeout: float = 300.0,
            ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]] = None,

            # Randomness
            seed: Optional[int] = None,
            deterministic_mode: bool = False,

            # Logging and callbacks
            log_destinations: Optional[Sequence[BaseLoggerBackend]] = None,
            callbacks: Sequence[Callback] = tuple(),

            # Checkpoint hparams
            checkpoint_loader: Optional[CheckpointLoaderHparams] = None,
            checkpoint_saver: Optional[CheckpointSaverHparams] = None,

            # Profiling
            profiler: Optional[ProfilerHparams] = None,

            # Subset parameters
            train_subset_num_batches: Optional[int] = None,
            eval_subset_num_batches: Optional[int] = None,

            # DeepSpeed
            deepspeed_hparams: Optional[DeepSpeedHparams] = None,

            # Optional config (ex. an hparams yaml file)
            config: Optional[Dict[str, Any]] = None):
        # surpressing GradScaler warnings as they are always created
        # self._use_grad_scaling() will raise a RuntimeError if grad scaling is not available when it is required
        warnings.filterwarnings(action="ignore", message="torch.cuda.amp.GradScaler")

        if isinstance(max_duration, str):
            max_duration = Time.from_timestring(max_duration)

        self.config = config

        self.deepspeed_hparams = deepspeed_hparams

        if not device:
            device = DeviceCPU() if not self.deepspeed_hparams is not None else DeviceGPU()
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
            dist.initialize_dist(device.dist_backend, datetime.timedelta(seconds=dist_timeout))
            if ddp_sync_strategy is None:
                self.ddp_sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC if not find_unused_parameters else DDPSyncStrategy.FORCED_SYNC
            else:
                self.ddp_sync_strategy = DDPSyncStrategy(ddp_sync_strategy)

        # TODO(#123): DeepSpeed still needs a precision context, but it's not completely clear how to
        # handle this with our version of Pytorch
        precision_context = self.device.precision_context if not self.deepspeed_enabled else cast(
            Callable[..., ContextManager], contextlib.nullcontext)

        if not isinstance(train_dataloader, DataSpec):
            train_dataloader = DataSpec(train_dataloader)
        if not isinstance(eval_dataloader, DataSpec):
            eval_dataloader = DataSpec(eval_dataloader)

        self._train_data_spec = train_dataloader
        self._eval_data_spec = eval_dataloader

        if eval_subset_num_batches is not None:
            try:
                eval_dataloader_len = len(eval_dataloader.dataloader)
            except (NotImplementedError, TypeError):
                pass
            else:
                if eval_subset_num_batches > eval_dataloader_len:
                    warnings.warn(
                        textwrap.dedent(
                            f"""SubsetNumBatchesWarning: The eval_subset_num_batches({eval_subset_num_batches})
                            is greater than the number of batches in the evaluation dataloader
                            ({len(eval_dataloader.dataloader)})"""))
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
            eval_dataloader=eval_dataloader.dataloader,
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

        assert isinstance(self.state.model, BaseMosaicModel)
        self.original_model = self.state.model  # TODO(ravi) -- update the state to add an original model helper

        self.checkpoint_saver = None
        if checkpoint_saver is not None:
            self.checkpoint_saver = checkpoint_saver.initialize_object()

        self.checkpoint_loader = None
        if checkpoint_loader is not None:
            self.checkpoint_loader = checkpoint_loader.initialize_object()

        # place the state, model in the proper devices, and initialize from a checkpoint if provided
        if self.deepspeed_enabled:
            import deepspeed

            assert self.deepspeed_hparams is not None
            deepspeed_config = self.deepspeed_hparams.initialize_object(self.state, self.grad_clip_norm)
            optimizer = ensure_tuple(self.state.optimizers)[0]
            (self.state.model, self.state.optimizers, _, _) = deepspeed.initialize(
                config=deepspeed_config,
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

    @classmethod
    def create_from_hparams(cls, hparams: TrainerHparams) -> Trainer:
        """Instantiate a Trainer using a `TrainerHparams` object.

        Args:
            hparams (TrainerHparams): The TrainerHparams object used to instantiate the trainer.

        Returns:
            A Trainer object initialized with the provided TrainerHparams.
        """

        hparams.validate()
        import composer
        logging.getLogger(composer.__name__).setLevel(hparams.log_level)

        # devices and systems
        device = hparams.device.initialize_object()

        seed = hparams.seed if hparams.seed else reproducibility.get_random_seed()
        # need to set seed before model initialization for determinism
        # don't need to set different seeds per process since only the rank 0 initialization is used
        reproducibility.seed_all(seed)

        model = hparams.model.initialize_object()
        algorithms = [x.initialize_object() for x in hparams.algorithms]

        # callbacks, loggers, and seed
        dict_config = hparams.to_dict()
        log_destinations = [x.initialize_object(config=dict_config) for x in hparams.loggers]
        callbacks = [x.initialize_object() for x in hparams.callbacks]

        if hparams.datadir is not None:
            hparams.train_dataset.datadir = hparams.datadir
            hparams.val_dataset.datadir = hparams.datadir

        train_device_batch_size = hparams.train_batch_size // dist.get_world_size()
        if hparams.train_dataset.shuffle and hparams.train_subset_num_batches is not None:
            warnings.warn(
                textwrap.dedent(f"""SubsetNumBatchesWarning: When specifying train_subset_num_batches,
            (set to {hparams.train_subset_num_batches}), train_datset.shuffle should be set to False. Otherwise,
            each training epoch may load a different subset of samples."""))
        train_data = hparams.train_dataset.initialize_object(train_device_batch_size, hparams.dataloader)

        eval_device_batch_size = hparams.eval_batch_size // dist.get_world_size()
        if hparams.val_dataset.shuffle and hparams.eval_subset_num_batches is not None:
            warnings.warn(
                textwrap.dedent(f"""SubsetNumBatchesWarning: When specifying eval_subset_num_batches,
            (set to {hparams.eval_subset_num_batches}), val_dataset.shuffle should be set to False. Otherwise,
            each evaluation epoch may load a different subset of samples."""))
        eval_data = hparams.val_dataset.initialize_object(eval_device_batch_size, hparams.dataloader)

        optimizers = hparams.optimizer.initialize_object(model.parameters())

        train_dataloader = train_data

        samples_per_epoch = None
        tokens_per_epoch = None

        if isinstance(train_dataloader, DataSpec):
            if train_dataloader.num_samples is not None:
                samples_per_epoch = train_dataloader.num_samples
                tokens_per_epoch = train_dataloader.num_tokens
            train_dataloader = train_dataloader.dataloader

        try:
            steps_per_epoch = len(train_dataloader)
        except (AttributeError, NotImplementedError):
            steps_per_epoch = None

        batch_size = None
        if train_dataloader.batch_size is not None:
            batch_size = train_dataloader.batch_size * dist.get_world_size()

        if samples_per_epoch is None and steps_per_epoch is not None and batch_size is not None:
            samples_per_epoch = steps_per_epoch * batch_size

        schedulers = [
            x.initialize_object(optimizer=optimizers,
                                max_training_duration=hparams.max_duration,
                                steps_per_epoch=steps_per_epoch,
                                samples_per_epoch=samples_per_epoch,
                                dataset_num_tokens=tokens_per_epoch) for x in ensure_warmup_last(hparams.schedulers)
        ]

        trainer = cls(
            model=model,
            train_dataloader=train_data,
            eval_dataloader=eval_data,
            max_duration=hparams.max_duration,
            algorithms=algorithms,
            optimizers=optimizers,
            schedulers=schedulers,

            # device
            device=device,

            # training hparams
            grad_accum=hparams.grad_accum,
            grad_clip_norm=hparams.grad_clip_norm,
            validate_every_n_batches=hparams.validate_every_n_batches,
            validate_every_n_epochs=hparams.validate_every_n_epochs,
            compute_training_metrics=hparams.compute_training_metrics,
            precision=hparams.precision,

            # dist hparams
            dist_timeout=hparams.dist_timeout,
            ddp_sync_strategy=hparams.ddp_sync_strategy,

            # Randomness
            seed=seed,
            deterministic_mode=hparams.deterministic_mode,

            # Callbacks and logging
            log_destinations=log_destinations,
            callbacks=callbacks,

            # Profiler
            profiler=hparams.profiler,

            # Checkpoint hparams
            checkpoint_loader=hparams.load_checkpoint,
            checkpoint_saver=hparams.save_checkpoint,

            # Subset parameters
            train_subset_num_batches=hparams.train_subset_num_batches,
            eval_subset_num_batches=hparams.eval_subset_num_batches,

            # DeepSpeed
            deepspeed_hparams=hparams.deepspeed,

            # Optional config
            config=hparams.to_dict())

        return trainer

    @property
    def deepspeed_enabled(self):
        return self.deepspeed_hparams is not None

    def fit(self):
        """Train and evaluate the model on the provided data."""
        try:
            self._train_loop()
        finally:
            self.engine.close()

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
        metrics = self.original_model.metrics(train=is_train)
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
            metric.set_dtype(torch.float32)  # type: ignore

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
        # spin the eval dataloader once to initialize its sampler deterministically
        # so it does not affect any other RNG reads
        if isinstance(self.state.eval_dataloader.sampler, torch.utils.data.DistributedSampler):
            self.state.eval_dataloader.sampler.set_epoch(0)
        for _ in self.state.eval_dataloader:
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
            train_metrics = self._get_metrics_as_collection(is_train=True)
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
                                total_loss = state.scaler.step(optimizer,
                                                               closure=lambda: self._train_batch(microbatches))
                            else:
                                total_loss = optimizer.step(closure=lambda: self._train_batch(microbatches).item())
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
                        dist.barrier()
                        full_loss = total_loss.cpu().item()
                        self.logger.metric_batch({'loss/train': full_loss / dist.get_world_size()})

                    if self.compute_training_metrics:
                        assert train_metrics is not None
                        self._compute_and_log_metrics(train_metrics, is_train=True, is_batch=True)

                    self.engine.run_event(Event.BATCH_END)

                    for scheduler in state.schedulers:
                        scheduler.step(interval='batch')  # type: ignore

                    state.timer.on_batch_complete(
                        samples=int(num_samples_in_batch.item()),
                        tokens=int(num_tokens_in_batch.item()),
                    )

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

            for scheduler in state.schedulers:
                scheduler.step(interval='epoch')  # type: ignore

            self.engine.run_event(Event.EPOCH_END)

            state.timer.on_epoch_complete()

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
        """Iterate over microbatches and compute the loss that will be used to step
        the optimizer.
        """
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

            metrics = self._get_metrics_as_collection(is_train=False)

            if isinstance(self.state.eval_dataloader.sampler, torch.utils.data.DistributedSampler):
                # The distributed sampler uses `set_epoch` to set the random seed
                # Because evaluation can run on each batch, we use the batch to seed the sampler
                # so each evaluation will get a proper shuffle.
                # The epoch provided to `set_epoch` need not be sequential, so this is fine.
                self.state.eval_dataloader.sampler.set_epoch(int(self.state.timer.batch))

            for state.batch in itertools.islice(state.eval_dataloader, self._eval_subset_num_batches):
                state.batch = self.device.batch_to_device(state.batch)
                state.batch = self._eval_data_spec.device_transforms(state.batch)
                state.batch_num_samples = self._eval_data_spec.get_num_samples_in_batch(state.batch)
                state.batch_num_tokens = self._eval_data_spec.get_num_tokens_in_batch(state.batch)

                if self.deepspeed_enabled:
                    state.batch = fix_batch_precision_for_deepspeed(state.batch, state.precision)

                self.engine.run_event(Event.EVAL_BATCH_START)

                self.engine.run_event(Event.EVAL_BEFORE_FORWARD)
                state.outputs, targets = self.original_model.validate(state.batch)
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

        We default to using closures unless AMP is enabled, in which case we only allow
        closures when using optimizers with the _step_supports_amp_closure flag.
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
