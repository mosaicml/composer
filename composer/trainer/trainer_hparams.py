# Copyright 2021 MosaicML. All Rights Reserved.

"""The :class:`~yahp.hparams.Hparams` used to construct the :class:`~composer.trainer.trainer.Trainer`."""

from __future__ import annotations

import logging
import os
import textwrap
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import yahp as hp

import composer
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import (CallbackHparams, GradMonitorHparams, LRMonitorHparams, MemoryMonitorHparams,
                                SpeedMonitorHparams)
from composer.core import Precision
from composer.core.types import JSON
from composer.datasets import DataLoaderHparams, DatasetHparams
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.evaluator import EvaluatorHparams
from composer.loggers import LoggerDestinationHparams, logger_registry
from composer.models import (BERTForClassificationHparams, BERTHparams, DeepLabV3Hparams, EfficientNetB0Hparams,
                             GPT2Hparams, MnistClassifierHparams, ModelHparams, ResNetCIFARHparams, ResNetHparams,
                             SSDHparams, TimmHparams, UnetHparams, ViTSmallPatch16Hparams)
from composer.optim import (AdamHparams, AdamWHparams, ConstantSchedulerHparams, CosineAnnealingSchedulerHparams,
                            CosineAnnealingWarmRestartsSchedulerHparams, CosineAnnealingWithWarmupSchedulerHparams,
                            DecoupledAdamWHparams, DecoupledSGDWHparams, ExponentialSchedulerHparams,
                            LinearSchedulerHparams, LinearWithWarmupSchedulerHparams, MultiStepSchedulerHparams,
                            MultiStepWithWarmupSchedulerHparams, OptimizerHparams, PolynomialSchedulerHparams,
                            RAdamHparams, RMSpropHparams, SchedulerHparams, SGDHparams, StepSchedulerHparams)
from composer.profiler.profiler_hparams import (ProfileScheduleHparams, TraceHandlerHparams,
                                                profiler_scheduler_registry, trace_handler_registory)
from composer.trainer.ddp import DDPSyncStrategy
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.trainer.trainer import Trainer
from composer.utils import dist, reproducibility
from composer.utils.object_store import ObjectStoreHparams

if TYPE_CHECKING:
    from composer.trainer.trainer import Trainer

__all__ = ["TrainerHparams"]

optimizer_registry = {
    "adam": AdamHparams,
    "adamw": AdamWHparams,
    "decoupled_adamw": DecoupledAdamWHparams,
    "radam": RAdamHparams,
    "sgd": SGDHparams,
    "decoupled_sgdw": DecoupledSGDWHparams,
    "rmsprop": RMSpropHparams,
}

scheduler_registry = {
    "step": StepSchedulerHparams,
    "multistep": MultiStepSchedulerHparams,
    "exponential": ExponentialSchedulerHparams,
    "linear_decay": LinearSchedulerHparams,
    "cosine_decay": CosineAnnealingSchedulerHparams,
    "cosine_warmrestart": CosineAnnealingWarmRestartsSchedulerHparams,
    "constant": ConstantSchedulerHparams,
    "polynomial": PolynomialSchedulerHparams,
    "multistep_with_warmup": MultiStepWithWarmupSchedulerHparams,
    "linear_decay_with_warmup": LinearWithWarmupSchedulerHparams,
    "cosine_decay_with_warmup": CosineAnnealingWithWarmupSchedulerHparams,
}

model_registry = {
    "unet": UnetHparams,
    "ssd": SSDHparams,
    "deeplabv3": DeepLabV3Hparams,
    "efficientnetb0": EfficientNetB0Hparams,
    "resnet_cifar": ResNetCIFARHparams,
    "resnet": ResNetHparams,
    "mnist_classifier": MnistClassifierHparams,
    "gpt2": GPT2Hparams,
    "bert": BERTHparams,
    "bert_classification": BERTForClassificationHparams,
    "timm": TimmHparams,
    "vit_small_patch16": ViTSmallPatch16Hparams
}

dataset_registry = get_dataset_registry()

algorithms_registry = get_algorithm_registry()

callback_registry = {
    "speed_monitor": SpeedMonitorHparams,
    "lr_monitor": LRMonitorHparams,
    "grad_monitor": GradMonitorHparams,
    "memory_monitor": MemoryMonitorHparams,
}

device_registry = {
    "gpu": GPUDeviceHparams,
    "cpu": CPUDeviceHparams,
}

evaluator_registry = {"evaluator": EvaluatorHparams}


@dataclass
class TrainerHparams(hp.Hparams):
    """Params for instantiating the :class:`.Trainer`.

    .. seealso:: The documentation for the :class:`.Trainer`.

    Args:
        model (ModelHparams): Hparams for constructing the model to train.

            .. seealso:: :mod:`composer.models` for models built into Composer.
        train_dataset (DatasetHparams): Hparams used to construct the dataset used for training.

            .. seealso:: :mod:`composer.datasets` for datasets built into Composer.
        train_batch_size (int): The optimization batch size to use for training. This is the total batch
            size that is used to produce a gradient for the optimizer update step.
        dataloader (DataLoaderHparams): Hparams used for constructing the dataloader which will be used
            for loading the train dataset and (if provided) the validation dataset.
        max_duration (str): The maximum duration to train as a str (e.g. ``1ep``, or ``10ba``).
            Will be converted to a :class:`~composer.core.Time` object.

            .. seealso:: :class:`~composer.core.Time` for more details on time construction.
        datadir (str, optional): Datadir to apply for both the training and validation datasets. If specified,
            it will override both ``train_dataset.datadir`` and ``val_dataset.datadir``. (default: ``None``)
        val_dataset (DatasetHparams, optional): Hparams for constructing the dataset used for evaluation.
            (default: ``None``)

            .. seealso:: :mod:`composer.datasets` for datasets built into Composer.
        eval_batch_size (int, optional): The batch size to use for evaluation. Must be provided if one of
            ``val_dataset`` or ``evaluators`` is set. (default: ``None``)
        evaluators (List[EvaluatorHparams], optional): Hparams for constructing evaluators to be used during the
            eval loop. Evaluators should be used when evaluating one or more specific metrics across one
            or more datasets. (default: ``None``)

            .. seealso:: :class:`~composer.core.evaluator.Evaluator` for more details on evaluators.
        algorithms (List[AlgorithmHparams], optional): The algorithms to use during training. (default: ``[]``)

            .. seealso:: :mod:`composer.algorithms` for the different algorithms built into Composer.
        optimizers (OptimizerHparams, optional): The hparams for constructing the optimizer. (default: ``None``)

            .. seealso:: :class:`.Trainer` for the default optimizer behavior when ``None`` is provided.

            .. seealso:: :mod:`composer.optim` for the different optimizers built into Composer.
        schedulers (List[SchedulerHparams], optional): The learning rate schedulers. (default: ``[]``).

            .. seealso:: :class:`.Trainer` for the default scheduler behavior when ``[]`` is provided.

            .. seealso:: :mod:`composer.optim.scheduler` for the different schedulers built into Composer.
        device (DeviceHparams): Hparams for constructing the device used for training.
            (default: ``CPUDeviceHparams``)
        grad_accum (int, optional): See :class:`.Trainer`.
        grad_clip_norm (float, optional): See :class:`.Trainer`.
        validate_every_n_batches (int, optional): See :class:`.Trainer`.
        validate_every_n_epochs (int, optional): See :class:`.Trainer`.
        compute_training_metrics (bool, optional): See :class:`.Trainer`.
        precision (Precision, optional): See :class:`.Trainer`.
        scale_schedule_ratio (float, optional): See :class:`.Trainer`.
        step_schedulers_every_batch (bool, optional): See :class:`.Trainer`.
        dist_timeout (float, optional): See :class:`.Trainer`.
        ddp_sync_strategy (DDPSyncStrategy, optional): See :class:`.Trainer`.
        seed (int, optional): See :class:`.Trainer`.
        deterministic_mode (bool, optional): See :class:`.Trainer`.
        run_name (str, optional): See :class:`.Trainer`.
        loggers (List[LoggerDestinationHparams], optional): Hparams for constructing the destinations
            to log to. (default: ``[]``)

            .. seealso:: :mod:`composer.loggers` for the different loggers built into Composer.
        log_level (str): The Python log level to use for log statements in the :mod:`composer`
            module. (default: ``INFO``)

            .. seealso:: The :mod:`logging` module in Python.
        callbacks (List[CallbackHparams], optional): Hparams to construct the callbacks to
            run during training. (default: ``[]``)

            .. seealso:: :mod:`composer.callbacks` for the different callbacks built into Composer.
        load_path (str, optional): See :class:`.Trainer`.
        load_object_store (ObjectStore, optional): See :class:`.Trainer`.
        load_weights_only (bool, optional): See :class:`.Trainer`.
        load_chunk_size (int, optional): See :class:`.Trainer`.
        save_folder (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_filename (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_latest_filename (str, optional): See
            :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_overwrite (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_weights_only (bool, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_interval (str, optional): See
            :class:`~composer.callbacks.callback_hparams.CheckpointSaverHparams`.
        save_num_checkpoints_to_keep (int, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        train_subset_num_batches (int, optional): See :class:`.Trainer`.
        eval_subset_num_batches (int, optional): See :class:`.Trainer`.
        deepspeed_config (Dict[str, JSON], optional): If set to a dict will be used for as the DeepSpeed
            config for training  (see :class:`.Trainer` for more details). If ``None`` will pass ``False``
            to the trainer for the ``deepspeed_config`` parameter signaling that DeepSpeed will not be used
            for training.
        prof_schedule (ProfileScheduleHparams, optional): Profile schedule hparams. Must be specified to enable the profiler.
        prof_event_handlers (List[TraceHandlerHparams], optional): See :class:`.Trainer`. Must be specified to enable the profiler.        prof_skip_first (int, optional): See :class:`.Trainer`.        prof_wait (int, optional): See :class:`.Trainer`.

        sys_prof_cpu (bool, optional): See :class:`.Trainer`.
        sys_prof_memory (bool, optional): See :class:`.Trainer`.
        sys_prof_disk (bool, optional): See :class:`.Trainer`.
        sys_prof_net (bool, optional): See :class:`.Trainer`.
        sys_prof_stats_thread_interval_seconds (float, optional): See :class:`.Trainer`.
        torch_prof_folder (str, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_filename (str, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_artifact_name (str, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_overwrite (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_use_gzip (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_record_shapes (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_profile_memory (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_with_stack (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_with_flops (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
        torch_prof_num_traces_to_keep (int, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_event_handlers`` are not specified.
    """

    hparams_registry = {  # type: ignore
        "algorithms": algorithms_registry,
        "optimizer": optimizer_registry,
        "schedulers": scheduler_registry,
        "loggers": logger_registry,
        "model": model_registry,
        "train_dataset": dataset_registry,
        "val_dataset": dataset_registry,
        "callbacks": callback_registry,
        "device": device_registry,
        "prof_trace_handlers": trace_handler_registory,
        "prof_schedule": profiler_scheduler_registry,
        "evaluators": evaluator_registry,
    }

    model: ModelHparams = hp.required(doc="model")

    # train data
    train_dataset: DatasetHparams = hp.required(doc="Training dataset hparams")
    train_batch_size: int = hp.required(
        doc="batch size for each optimization step, across all devices and gradient accumulations.")
    dataloader: DataLoaderHparams = hp.required(doc="dataloader hparams")

    # duration
    max_duration: str = hp.required(doc="Time string for the maximum training duration (e.g., 90ep)")

    # datadir
    datadir: Optional[str] = hp.optional(doc=textwrap.dedent("""\
        Datadir to apply for both the training and validation datasets. If specified,
        it will override train_dataset.datadir and val_dataset.datadir."""),
                                         default=None)

    # eval
    val_dataset: Optional[DatasetHparams] = hp.optional(doc="Validation dataset hparams", default=None)
    eval_batch_size: Optional[int] = hp.optional(doc="batch size to use for each evaluation step", default=None)
    evaluators: Optional[List[EvaluatorHparams]] = hp.optional(doc="Evaluators", default=None)

    # training algos
    algorithms: List[AlgorithmHparams] = hp.optional(doc="Algorithms to employ", default_factory=list)
    optimizer: Optional[OptimizerHparams] = hp.optional(doc="Optimizer to use", default=None)
    schedulers: List[SchedulerHparams] = hp.optional(doc="Scheduler sequence", default_factory=list)

    # device
    device: DeviceHparams = hp.optional(doc="Device Parameters", default_factory=CPUDeviceHparams)

    # training hparams
    grad_accum: Union[int, str] = hp.optional(textwrap.dedent("""\
        Determines the number of microbatches to split a per-gpu batch into,
        used to compensate for low-memory-capacity devices. If set to auto, 
        dynamically increases grad_accum if microbatch size is too large for
        GPU. Defaults to ``1``"""),
                                              default=1)
    grad_clip_norm: Optional[float] = hp.optional(
        default=None, doc='the norm to clip gradient magnitudes to. Default: None (no clip)')
    validate_every_n_epochs: int = hp.optional(
        doc="Validate every N epochs. Set to -1 to never validate on a epochwise frequency. Defaults to 1", default=1)
    validate_every_n_batches: int = hp.optional(
        doc="Validate every N batches. Set to -1 to never validate on a batchwise frequency. Defaults to -1.",
        default=-1)
    compute_training_metrics: bool = hp.optional(doc="Log validation metrics on training data", default=False)
    precision: Precision = hp.optional(doc="Precision to use for training", default=Precision.AMP)
    scale_schedule_ratio: float = hp.optional(
        doc="Ratio by which to scale the training duration and learning rate schedules.", default=1.0)
    step_schedulers_every_batch: bool = hp.optional(
        doc="Whether schedulers will update after every optimizer step (True), or every epoch (False).", default=True)

    # dist hparams
    dist_timeout: float = hp.optional(doc="Timeout, in seconds, for initializing the dsitributed process group.",
                                      default=15.0)
    ddp_sync_strategy: Optional[DDPSyncStrategy] = hp.optional(doc=textwrap.dedent("""\
            The strategy for synchronizing DDP. Default value ``None`` causes the
            trainer to auto-select a value depending on what algorithms are used."""),
                                                               default=None)

    # randomness
    seed: Optional[int] = hp.optional(default=None, doc="random seed to set")
    deterministic_mode: bool = hp.optional(textwrap.dedent("""\
        Run the model deterministically. Experimental. Performance
        degradations expected. Certain Torch modules may not have
        deterministic implementations, which will result in a crash."""),
                                           default=False)

    # logging and callbacks
    run_name: Optional[str] = hp.optional("Experiment name", default=None)
    loggers: List[LoggerDestinationHparams] = hp.optional(doc="loggers to use", default_factory=list)
    log_level: str = hp.optional(doc="Python loglevel to use composer", default="INFO")
    callbacks: List[CallbackHparams] = hp.optional(doc="Callback hparams", default_factory=list)

    # load checkpoint
    load_path: Optional[str] = hp.optional(doc=textwrap.dedent("""\
        If specified, the path to an existing checkpoint file
        (if the checkpoint is on the local disk) or the object name for the checkpoint
        (if the checkpoint is in a cloud bucket). Set to None (the default) to skip loading from a checkpoint."""),
                                           default=None)
    load_object_store: Optional[ObjectStoreHparams] = hp.optional(doc=textwrap.dedent("""\
        If the checkpoint is in an object store (i.e. AWS S3 or Google Cloud Storage), the parameters for
        connecting to the cloud provider object store. Otherwise, if the checkpoint is a local filepath,
        leave blank. This parameter has no effect if `load_path` is not specified."""),
                                                                  default=None)
    load_weights_only: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to only load the weights from the model.
        This parameter has no effect if `load_path`is not specified."""),
                                          default=False)
    load_strict_model_weights: bool = hp.optional(doc=textwrap.dedent("""\
        Ensure that the set of checkpoint weights in the checkpoint and model must exactly match.
        This parameter has no effect if `load_path` is not specified."""),
                                                  default=False)

    load_chunk_size: int = hp.optional(doc=textwrap.dedent("""\
        Chunk size (in bytes) to use when downloading checkpoints.
        This parameter has no effect if `load_path` is not specified or it is a local file path."""),
                                       default=1_048_576)
    load_progress_bar: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to show a progress bar when downloading checkpoints.
        This parameter has no effect if `load_path` is not specified or it is a local file path."""),
                                          default=True)

    # save checkpoint
    save_folder: Optional[str] = hp.optional(doc="Checkpoint folder format string.", default=None)
    save_filename: str = hp.optional("Checkpoint name format string.", default="ep{epoch}-ba{batch}-rank{rank}")
    save_artifact_name: str = hp.optional("Checkpoint artifact name format",
                                          default="{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}")
    save_latest_filename: str = hp.optional("Latest checkpoint symlink format string.", default="latest-rank{rank}")
    save_overwrite: bool = hp.optional("Whether to override existing checkpoints.", default=False)
    save_weights_only: bool = hp.optional("Whether to save only checkpoint weights", default=False)
    save_interval: str = hp.optional(textwrap.dedent("""\
        Checkpoint interval or path to a `(State, Event) -> bool` function
        returning whether a checkpoint should be saved."""),
                                     default="1ep")
    save_num_checkpoints_to_keep: int = hp.optional(
        "Number of checkpoints to persist locally. Set to -1 to never delete checkpoints.",
        default=-1,
    )

    # subset parameters
    train_subset_num_batches: Optional[int] = hp.optional(
        "If specified, finish every epoch early after training on this many batches.", default=None)
    eval_subset_num_batches: Optional[int] = hp.optional("If specified, stop each evaluation after this many batches.",
                                                         default=None)

    # DeepSpeed
    deepspeed: Optional[Dict[str, JSON]] = hp.optional(doc="Configuration for DeepSpeed.", default=None)

    # profiling
    prof_trace_handlers: List[TraceHandlerHparams] = hp.optional(doc=textwrap.dedent("""\
        Trace event handlers. Must be specified to activate the profiler."""),
                                                                 default_factory=list)
    prof_schedule: Optional[ProfileScheduleHparams] = hp.optional(doc="Profile scheduler hparams", default=None)

    sys_prof_cpu: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record cpu statistics.  Ignored if `prof_trace_handlers` is not specified."""),
                                     default=True)
    sys_prof_memory: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record memory statistics.  Ignored if `prof_trace_handlers` is not specified."""),
                                        default=False)
    sys_prof_disk: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record disk statistics.  Ignored if `prof_trace_handlers` is not specified."""),
                                      default=False)
    sys_prof_net: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record network statistics.  Ignored if `prof_trace_handlers` is not specified."""),
                                     default=False)
    sys_prof_stats_thread_interval_seconds: float = hp.optional(doc=textwrap.dedent("""\
        Interval to record stats, in seconds.  Ignored if `prof_trace_handlers` is not specified."""),
                                                                default=0.5)

    torch_prof_folder: str = hp.optional('Torch profiler folder format', default='{run_name}/torch_traces')
    torch_prof_filename: str = hp.optional(
        'Torch profiler filename format',
        default='rank{rank}.{batch}.pt.trace.json',
    )
    torch_prof_artifact_name: str = hp.optional(
        'Torch profiler artifact name format',
        default='{run_name}/torch_traces/rank{rank}.{batch}.pt.trace.json',
    )
    torch_prof_overwrite: bool = hp.optional('Torch profiler overwrite', default=False)
    torch_prof_use_gzip: bool = hp.optional('Torch profiler use gzip', default=False)
    torch_prof_num_traces_to_keep: int = hp.optional('Torch profiler num traces to keep', default=-1)
    torch_prof_record_shapes: bool = hp.optional(
        "Whether to record tensor shapes. Ignored if `prof_trace_handlers` is not specified.",
        default=False,
    )
    torch_prof_profile_memory: bool = hp.optional(
        "Track tensor memory allocations and frees. Ignored if `prof_trace_handlers` is not specified.",
        default=True,
    )
    torch_prof_with_stack: bool = hp.optional(
        "Record stack information. Ignored if `prof_trace_handlers` is not specified.",
        default=False,
    )
    torch_prof_with_flops: bool = hp.optional(
        "Estimate flops for operators. Ignored if `prof_trace_handlers` is not specified.",
        default=True,
    )

    def validate(self):
        super().validate()

        if self.deepspeed is not None:
            self.deepspeed["steps_per_print"] = cast(int, self.deepspeed.get("steps_per_print", 1e20))

            if "zero_optimization" in self.deepspeed:
                zero_stage = cast(dict, self.deepspeed["zero_optimization"]).get("stage", 0)
            else:
                zero_stage = 0

            if self.deterministic_mode and zero_stage > 0:
                raise ValueError("Deepspeed with zero stage > 0 is not compatible with deterministic mode")

            if isinstance(self.device, CPUDeviceHparams):
                raise ValueError("Training on CPUs is not supported with DeepSpeed.")

        elif self.precision == Precision.FP16:
            raise ValueError("FP16 precision is only supported when training with DeepSpeed.")

        world_size = dist.get_world_size()

        if self.train_batch_size % world_size != 0:
            raise ValueError(
                f"Batch size ({self.train_batch_size}) not divisible by the total number of processes ({world_size}).")

        val_dataset_exists = self.val_dataset is not None
        evaluators_exist = self.evaluators is not None and len(self.evaluators) > 0
        if val_dataset_exists and evaluators_exist:
            raise ValueError("Either val_dataset or evaluators should be set, but not both.")

        if (val_dataset_exists or evaluators_exist) and self.eval_batch_size is None:
            raise ValueError("eval_batch_size must be specified if val_dataset or evaluators are specified.")

        if self.eval_batch_size is not None and self.eval_batch_size % world_size != 0:
            raise ValueError(
                f"Eval batch size ({self.eval_batch_size}) not divisible by the total number of processes ({world_size})."
            )

        if self.scale_schedule_ratio <= 0:
            raise ValueError("scale_schedule_ratio must be a positive value.")

        if (isinstance(self.grad_accum, str) and self.grad_accum != "auto") or (isinstance(self.grad_accum, int) and
                                                                                self.grad_accum < 1):
            raise ValueError('grad_accum must be "auto" or an int greater than or equal to 1')

    def initialize_object(self) -> Trainer:
        self.validate()
        import composer
        logging.getLogger(composer.__name__).setLevel(self.log_level)

        # devices and systems
        device = self.device.initialize_object()

        seed = self.seed if self.seed else reproducibility.get_random_seed()
        # need to set seed before model initialization for determinism
        # don't need to set different seeds per process since only the rank 0 initialization is used
        # Algorithms should not use the `seed` on `__init__` but rather on `Event.INIT`, which occurs
        # after the seed was properly distributed across ranks to ensure checkpoint compatibility
        reproducibility.seed_all(seed)

        model = self.model.initialize_object()
        algorithms = [x.initialize_object() for x in self.algorithms]

        # callbacks, loggers, and seed
        loggers = [x.initialize_object() for x in self.loggers]
        callbacks = [x.initialize_object() for x in self.callbacks]

        if self.datadir is not None:
            self.train_dataset.datadir = self.datadir
            if self.val_dataset is not None:
                self.val_dataset.datadir = self.datadir

        train_device_batch_size = self.train_batch_size // dist.get_world_size()
        if self.train_dataset.shuffle and self.train_subset_num_batches is not None:
            warnings.warn(
                textwrap.dedent(f"""\
                SubsetNumBatchesWarning: When specifying train_subset_num_batches,
                (set to {self.train_subset_num_batches}), train_datset.shuffle should be set to False. Otherwise,
                each training epoch may load a different subset of samples."""))
        train_data = self.train_dataset.initialize_object(train_device_batch_size, self.dataloader)

        eval_device_batch_size = (self.eval_batch_size or 0) // dist.get_world_size()

        eval_dataloader = None
        if self.val_dataset is not None:
            if self.val_dataset.shuffle and self.eval_subset_num_batches is not None:
                warnings.warn(
                    textwrap.dedent(f"""\
                        SubsetNumBatchesWarning: When specifying eval_subset_num_batches,
                        (set to {self.eval_subset_num_batches}), val_dataset.shuffle should be
                        set to False. Otherwise, each evaluation epoch may load a different
                        subset of samples."""))
            eval_dataloader = self.val_dataset.initialize_object(eval_device_batch_size, self.dataloader)

        if self.evaluators is not None and len(self.evaluators) > 0:
            eval_dataloader = [
                evaluator.initialize_object(model, eval_device_batch_size, self.dataloader)
                for evaluator in self.evaluators
            ]
            for evaluator in self.evaluators:
                if evaluator.eval_dataset.shuffle and self.eval_subset_num_batches is not None:
                    warnings.warn(
                        textwrap.dedent(f"""SubsetNumBatchesWarning: When specifying eval_subset_num_batches,
                    (set to {self.eval_subset_num_batches}), evaluator.dataloader.shuffle (for Evaluator: "{evaluator.label}") should be set to False. Otherwise,
                    each evaluation epoch may load a different subset of samples."""))

        optimizer = self.optimizer.initialize_object(model.parameters()) if self.optimizer is not None else None
        schedulers = [scheduler.initialize_object() for scheduler in self.schedulers]

        deepspeed_config = self.deepspeed if self.deepspeed is not None else False

        trainer = Trainer(
            model=model,
            train_dataloader=train_data,
            eval_dataloader=eval_dataloader,
            max_duration=self.max_duration,
            algorithms=algorithms,
            optimizers=optimizer,
            schedulers=schedulers,

            # device
            device=device,

            # training hparams
            grad_accum=self.grad_accum,
            grad_clip_norm=self.grad_clip_norm,
            validate_every_n_batches=self.validate_every_n_batches,
            validate_every_n_epochs=self.validate_every_n_epochs,
            compute_training_metrics=self.compute_training_metrics,
            precision=self.precision,
            scale_schedule_ratio=self.scale_schedule_ratio,
            step_schedulers_every_batch=self.step_schedulers_every_batch,

            # dist hparams
            dist_timeout=self.dist_timeout,
            ddp_sync_strategy=self.ddp_sync_strategy,

            # Randomness
            seed=seed,
            deterministic_mode=self.deterministic_mode,

            # Callbacks and logging
            run_name=self.run_name,
            loggers=loggers,
            callbacks=callbacks,

            # Profiler
            prof_trace_handlers=[x.initialize_object() for x in self.prof_trace_handlers],
            prof_schedule=None if self.prof_schedule is None else self.prof_schedule.initialize_object(),

            # System profiler
            sys_prof_cpu=self.sys_prof_cpu,
            sys_prof_memory=self.sys_prof_memory,
            sys_prof_disk=self.sys_prof_disk,
            sys_prof_net=self.sys_prof_net,
            sys_prof_stats_thread_interval_seconds=self.sys_prof_stats_thread_interval_seconds,

            # Torch profiler
            torch_prof_folder=self.torch_prof_folder,
            torch_prof_filename=self.torch_prof_filename,
            torch_prof_artifact_name=self.torch_prof_artifact_name,
            torch_prof_overwrite=self.torch_prof_overwrite,
            torch_prof_use_gzip=self.torch_prof_use_gzip,
            torch_prof_num_traces_to_keep=self.torch_prof_num_traces_to_keep,
            torch_prof_record_shapes=self.torch_prof_record_shapes,
            torch_prof_profile_memory=self.torch_prof_profile_memory,
            torch_prof_with_stack=self.torch_prof_with_flops,
            torch_prof_with_flops=self.torch_prof_with_flops,

            # Checkpoint parameters
            load_path=self.load_path,
            load_object_store=None if self.load_object_store is None else self.load_object_store.initialize_object(),
            load_weights_only=self.load_weights_only,
            load_strict=self.load_strict_model_weights,
            load_chunk_size=self.load_chunk_size,
            load_progress_bar=self.load_progress_bar,
            save_folder=self.save_folder,
            save_overwrite=self.save_overwrite,
            save_filename=self.save_filename,
            save_latest_filename=self.save_latest_filename,
            save_artifact_name=self.save_artifact_name,
            save_interval=self.save_interval,
            save_weights_only=self.save_weights_only,
            save_num_checkpoints_to_keep=self.save_num_checkpoints_to_keep,

            # Subset parameters
            train_subset_num_batches=self.train_subset_num_batches,
            eval_subset_num_batches=self.eval_subset_num_batches,

            # DeepSpeed
            deepspeed_config=deepspeed_config,
        )

        return trainer

    @classmethod
    def load(cls, model: str) -> TrainerHparams:
        model_hparams_file = os.path.join(
            os.path.dirname(composer.__file__),
            "yamls",
            "models",
            f"{model}.yaml",
        )
        trainer_hparams = TrainerHparams.create(model_hparams_file, cli_args=False)
        assert isinstance(trainer_hparams, TrainerHparams), "trainer hparams should return an instance of self"
        return trainer_hparams
