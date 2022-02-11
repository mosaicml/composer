# Copyright 2021 MosaicML. All Rights Reserved.

"""Example usage and definition of hparams."""
from __future__ import annotations

import logging
import os
import textwrap
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, cast

import yahp as hp

import composer
from composer import datasets
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import (CallbackHparams, GradMonitorHparams, LRMonitorHparams, MemoryMonitorHparams,
                                RunDirectoryUploaderHparams, SpeedMonitorHparams)
from composer.core import DataSpec
from composer.core.types import JSON, Precision
from composer.datasets import DataloaderHparams
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.evaluator import EvaluatorHparams
from composer.loggers import (FileLoggerHparams, InMemoryLoggerHaparms, LoggerCallbackHparams, TQDMLoggerHparams,
                              WandBLoggerHparams)
from composer.models import (BERTForClassificationHparams, BERTHparams, CIFARResNet9Hparams, CIFARResNetHparams,
                             DeepLabV3Hparams, EfficientNetB0Hparams, GPT2Hparams, MnistClassifierHparams, ModelHparams,
                             ResNetHparams, TimmHparams, UnetHparams)
from composer.models.resnet20_cifar10.resnet20_cifar10_hparams import CIFARResNet20Hparams
from composer.optim import (AdamHparams, AdamWHparams, DecoupledAdamWHparams, DecoupledSGDWHparams, OptimizerHparams,
                            RAdamHparams, RMSPropHparams, SchedulerHparams, SGDHparams, scheduler)
from composer.profiler.profiler_hparams import JSONTraceHandlerHparams, ProfilerEventHandlerHparams
from composer.trainer.ddp import DDPSyncStrategy
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.trainer.trainer import Trainer
from composer.utils import dist, reproducibility
from composer.utils.object_store import ObjectStoreProviderHparams

if TYPE_CHECKING:
    from composer.trainer.trainer import Trainer

optimizer_registry = {
    "adam": AdamHparams,
    "adamw": AdamWHparams,
    "decoupled_adamw": DecoupledAdamWHparams,
    "radam": RAdamHparams,
    "sgd": SGDHparams,
    "decoupled_sgdw": DecoupledSGDWHparams,
    "rmsprop": RMSPropHparams,
}

scheduler_registry = {
    "step": scheduler.StepLRHparams,
    "multistep": scheduler.MultiStepLRHparams,
    "exponential": scheduler.ExponentialLRHparams,
    "linear_decay": scheduler.LinearLRHparams,
    "cosine_decay": scheduler.CosineAnnealingLRHparams,
    "cosine_warmrestart": scheduler.CosineAnnealingWarmRestartsHparams,
    "constant": scheduler.ConstantLRHparams,
    "polynomial": scheduler.PolynomialLRHparams,
    "multistep_with_warmup": scheduler.MultiStepWithWarmupLRHparams,
    "linear_decay_with_warmup": scheduler.LinearWithWarmupLRHparams,
    "cosine_decay_with_warmup": scheduler.CosineAnnealingWithWarmupLRHparams,
}

model_registry = {
    "unet": UnetHparams,
    "deeplabv3": DeepLabV3Hparams,
    "efficientnetb0": EfficientNetB0Hparams,
    "resnet56_cifar10": CIFARResNetHparams,
    "resnet20_cifar10": CIFARResNet20Hparams,
    "resnet9_cifar10": CIFARResNet9Hparams,
    "resnet": ResNetHparams,
    "mnist_classifier": MnistClassifierHparams,
    "gpt2": GPT2Hparams,
    "bert": BERTHparams,
    "bert_classification": BERTForClassificationHparams,
    "timm": TimmHparams
}

dataset_registry = get_dataset_registry()

algorithms_registry = get_algorithm_registry()

callback_registry = {
    "speed_monitor": SpeedMonitorHparams,
    "lr_monitor": LRMonitorHparams,
    "grad_monitor": GradMonitorHparams,
    "memory_monitor": MemoryMonitorHparams,
    "run_directory_uploader": RunDirectoryUploaderHparams,
}

logger_registry = {
    "file": FileLoggerHparams,
    "wandb": WandBLoggerHparams,
    "tqdm": TQDMLoggerHparams,
    "in_memory": InMemoryLoggerHaparms,
}

device_registry = {
    "gpu": GPUDeviceHparams,
    "cpu": CPUDeviceHparams,
}

prof_event_handlers_registry = {"json": JSONTraceHandlerHparams}


@dataclass
class TrainerHparams(hp.Hparams):
    """Params for the :class:`Trainer`.

    See the documentation for the :class:`Trainer`.
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
        "prof_event_handlers": prof_event_handlers_registry,
    }

    device: DeviceHparams = hp.required(doc="Device Parameters")
    train_dataset: datasets.DatasetHparams = hp.required(doc="Training dataset hparams")

    optimizer: OptimizerHparams = hp.required(doc="Optimizer to use")

    model: ModelHparams = hp.required(doc="model")
    loggers: List[LoggerCallbackHparams] = hp.required(doc="loggers to use")

    max_duration: str = hp.required(doc="Time string for the maximum training duration (e.g., 90ep)")

    train_batch_size: int = hp.required(
        doc="batch size for each optimization step, across all devices and gradient accumulations.")

    eval_batch_size: int = hp.required(doc="batch size to use for each evaluation step")

    dataloader: DataloaderHparams = hp.required(doc="dataloader hparams")

    grad_accum: int = hp.optional(textwrap.dedent("""\
        Determines the number of microbatches to split a per-gpu batch into,
        used to compensate for low-memory-capacity devices."""),
                                  default=1)
    precision: Precision = hp.optional(doc="Precision to use for training", default=Precision.AMP)

    val_dataset: Optional[datasets.DatasetHparams] = hp.optional(doc="Validation dataset hparams", default=None)

    evaluators: Optional[List[EvaluatorHparams]] = hp.optional(doc="Evaluators", default_factory=list)

    dist_timeout: float = hp.optional(doc="Timeout, in seconds, for initializing the dsitributed process group.",
                                      default=15.0)
    ddp_sync_strategy: Optional[DDPSyncStrategy] = hp.optional(doc=textwrap.dedent("""\
            The strategy for synchronizing DDP. Default value ``None`` causes the
            trainer to auto-select a value depending on what algorithms are used."""),
                                                               default=None)

    deepspeed: Optional[Dict[str, JSON]] = hp.optional(doc="Configuration for DeepSpeed.", default=None)

    grad_clip_norm: Optional[float] = hp.optional(
        default=None, doc='the norm to clip gradient magnitudes to. Default: None (no clip)')

    algorithms: List[AlgorithmHparams] = hp.optional(doc="Algorithms to employ", default_factory=list)
    schedulers: List[SchedulerHparams] = hp.optional(doc="Scheduler sequence", default_factory=list)
    seed: Optional[int] = hp.optional(default=None, doc="random seed to set")
    validate_every_n_epochs: int = hp.optional(
        doc="Validate every N epochs. Set to -1 to never validate on a epochwise frequency. Defaults to 1", default=1)
    validate_every_n_batches: int = hp.optional(
        doc="Validate every N batches. Set to -1 to never validate on a batchwise frequency. Defaults to -1.",
        default=-1)
    scale_schedule_ratio: float = hp.optional(
        doc="Ratio by which to scale the training duration and learning rate schedules.", default=1.0)
    use_stepwise_schedulers: bool = hp.optional(
        doc="Whether schedulers will update after every optimizer step (True), or every epoch (False).", default=True)
    callbacks: List[CallbackHparams] = hp.optional(doc="Callback hparams", default_factory=list)

    load_path: Optional[str] = hp.optional(doc=textwrap.dedent("""\
        If specified, the path to an existing checkpoint file
        (if the checkpoint is on the local disk) or the object name for the checkpoint
        (if the checkpoint is in a cloud bucket). Set to None (the default) to skip loading from a checkpoint."""),
                                           default=None)
    load_object_store: Optional[ObjectStoreProviderHparams] = hp.optional(doc=textwrap.dedent("""\
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

    save_folder: Optional[str] = hp.optional(doc=textwrap.dedent(f"""\
        Folder to save checkpoint files, relative to the run directory.
        Defaults to None, meaning checkpoints will not be saved."""),
                                             default=None)
    save_interval: str = hp.optional(doc=textwrap.dedent("""\
        The time string interval representing how frequently checkpoints should be saved.
        For example, set to "1ep" to save checkpoints every epoch, or "10ba"
        to save checkpoints every 10 batches.
        This parameter has no effect if `save_folder` is not specified."""),
                                     default="1ep")

    save_compression: str = hp.optional(doc=textwrap.dedent("""\
        Compression algorithm to run on checkpoints. Can be `gzip`, `bzip2`,
        `lzma`, or left blank for no compression.  (default: ``""`` for no compression)."""),
                                        default="")

    train_subset_num_batches: Optional[int] = hp.optional(
        "If specified, finish every epoch early after training on this many batches.", default=None)
    eval_subset_num_batches: Optional[int] = hp.optional("If specified, stop each evaluation after this many batches.",
                                                         default=None)

    deterministic_mode: bool = hp.optional(textwrap.dedent("""\
        Run the model deterministically. Experimental. Performance
        degradations expected. Certain Torch modules may not have
        deterministic implementations, which will result in a crash."""),
                                           default=False)

    compute_training_metrics: bool = hp.optional(doc="Log validation metrics on training data", default=False)
    log_level: str = hp.optional(doc="Python loglevel to use composer", default="INFO")
    datadir: Optional[str] = hp.optional(doc=textwrap.dedent("""\
        Datadir to apply for both the training and validation datasets. If specified,
        it will override train_dataset.datadir and val_dataset.datadir"""),
                                         default=None)

    profiler_trace_file: Optional[str] = hp.optional(doc=textwrap.dedent("""\
        Name of the trace file, relative to the run directory.  Must be specified to activate the profiler."""),
                                                     default=None)
    prof_event_handlers: List[ProfilerEventHandlerHparams] = hp.optional(
        doc=textwrap.dedent("""\
        Trace event handler.  Ignored if `profiler_trace_file` is not specified."""),
        default_factory=lambda: [JSONTraceHandlerHparams()])
    prof_skip_first: int = hp.optional(doc=textwrap.dedent("""\
        Number of batches to skip at epoch start.  Ignored if `profiler_trace_file` is not specified."""),
                                       default=0)
    prof_wait: int = hp.optional(doc=textwrap.dedent("""\
        Number of batches to skip at the beginning of each cycle.  Ignored if `profiler_trace_file` is not specified."""
                                                    ),
                                 default=0)
    prof_warmup: int = hp.optional(doc=textwrap.dedent("""\
        Number of warmup batches in a cycle.  Ignored if `profiler_trace_file` is not specified."""),
                                   default=1)
    prof_active: int = hp.optional(doc=textwrap.dedent("""\
        Number of batches to profile in a cycle.  Ignored if `profiler_trace_file` is not specified."""),
                                   default=4)
    prof_repeat: int = hp.optional(doc=textwrap.dedent("""\
        Maximum number of profiling cycle repetitions per epoch (0 for no maximum).  Ignored if `profiler_trace_file` is not specified."""
                                                      ),
                                   default=1)
    sys_prof_cpu: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record cpu statistics.  Ignored if `profiler_trace_file` is not specified."""),
                                     default=True)
    sys_prof_memory: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record memory statistics.  Ignored if `profiler_trace_file` is not specified."""),
                                        default=False)
    sys_prof_disk: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record disk statistics.  Ignored if `profiler_trace_file` is not specified."""),
                                      default=False)
    sys_prof_net: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record network statistics.  Ignored if `profiler_trace_file` is not specified."""),
                                     default=False)
    sys_prof_stats_thread_interval_seconds: float = hp.optional(doc=textwrap.dedent("""\
        Interval to record stats, in seconds.  Ignored if `profiler_trace_file` is not specified."""),
                                                                default=0.5)
    torch_profiler_trace_dir: Optional[str] = hp.optional(doc=textwrap.dedent("""\
        Directory to store trace results relative to the run directory.  Must be specified to activate the Torch profiler. 
        Ignored if ``profiler_trace_file`` is not specified."""),
                                                          default=None)
    torch_prof_use_gzip: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to use gzip for trace.  
        Ignored if ``torch_profiler_trace_dir`` and `profiler_trace_file` are not specified."""),
                                            default=False)

    torch_prof_record_shapes: bool = hp.optional(doc=textwrap.dedent("""\
        Whether to record tensor shapes.  
        Ignored if ``torch_profiler_trace_dir`` and `profiler_trace_file` are not specified."""),
                                                 default=False)
    torch_prof_profile_memory: bool = hp.optional(doc=textwrap.dedent("""\
        Track tensor memory allocations and frees.  
        Ignored if ``torch_profiler_trace_dir`` and `profiler_trace_file` are not specified."""),
                                                  default=True)
    torch_prof_with_stack: bool = hp.optional(doc=textwrap.dedent("""\
        Record stack information.  
        Ignored if ``torch_profiler_trace_dir`` and `profiler_trace_file` are not specified."""),
                                              default=False)
    torch_prof_with_flops: bool = hp.optional(doc=textwrap.dedent("""\
        Estimate flops for operators.  
        Ignored if ``torch_profiler_trace_dir`` and `profiler_trace_file` are not specified."""),
                                              default=True)

    def validate(self):
        super().validate()

        if self.deepspeed is not None:
            zero_stage = cast(int, self.deepspeed.get("zero_stage", 0))

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

        if self.eval_batch_size % world_size != 0:
            raise ValueError(
                f"Eval batch size ({self.eval_batch_size}) not divisible by the total number of processes ({world_size})."
            )

        if self.evaluators is not None and len(self.evaluators) > 0 and self.val_dataset is not None:
            raise ValueError(
                "val_dataset and evaluators shouldn't both be specified. Only one can be passed in to the trainer.")

        if self.scale_schedule_ratio <= 0:
            raise ValueError("scale_schedule_ratio must be a positive value.")

    def initialize_object(self) -> Trainer:
        self.validate()
        import composer
        logging.getLogger(composer.__name__).setLevel(self.log_level)

        # devices and systems
        device = self.device.initialize_object()

        seed = self.seed if self.seed else reproducibility.get_random_seed()
        # need to set seed before model initialization for determinism
        # don't need to set different seeds per process since only the rank 0 initialization is used
        reproducibility.seed_all(seed)

        model = self.model.initialize_object()
        algorithms = [x.initialize_object() for x in self.algorithms]

        # callbacks, loggers, and seed
        dict_config = self.to_dict()
        loggers = [x.initialize_object(config=dict_config) for x in self.loggers]
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

        eval_device_batch_size = self.eval_batch_size // dist.get_world_size()
        if self.val_dataset is not None and self.evaluators is not None and len(self.evaluators) > 0:
            raise ValueError("Either val_dataset or evaluators should be set, but not both.")

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

        optimizers = self.optimizer.initialize_object(model.parameters())

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

        schedulers = [scheduler.initialize_object() for scheduler in self.schedulers]

        trainer = Trainer(
            model=model,
            train_dataloader=train_data,
            eval_dataloader=eval_dataloader,
            max_duration=self.max_duration,
            algorithms=algorithms,
            optimizers=optimizers,
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
            use_stepwise_schedulers=self.use_stepwise_schedulers,

            # dist hparams
            dist_timeout=self.dist_timeout,
            ddp_sync_strategy=self.ddp_sync_strategy,

            # Randomness
            seed=seed,
            deterministic_mode=self.deterministic_mode,

            # Callbacks and logging
            loggers=loggers,
            callbacks=callbacks,

            # Profiler
            profiler_trace_file=self.profiler_trace_file,
            prof_event_handlers=[x.initialize_object() for x in self.prof_event_handlers],
            prof_skip_first=self.prof_skip_first,
            prof_wait=self.prof_wait,
            prof_warmup=self.prof_warmup,
            prof_active=self.prof_active,
            prof_repeat=self.prof_repeat,
            sys_prof_cpu=self.sys_prof_cpu,
            sys_prof_memory=self.sys_prof_memory,
            sys_prof_disk=self.sys_prof_disk,
            sys_prof_net=self.sys_prof_net,
            sys_prof_stats_thread_interval_seconds=self.sys_prof_stats_thread_interval_seconds,
            torch_profiler_trace_dir=self.torch_profiler_trace_dir,
            torch_prof_use_gzip=self.torch_prof_use_gzip,
            torch_prof_record_shapes=self.torch_prof_record_shapes,
            torch_prof_profile_memory=self.torch_prof_profile_memory,
            torch_prof_with_stack=self.torch_prof_with_flops,
            torch_prof_with_flops=self.torch_prof_with_flops,

            # Checkpoint parameters
            load_path=self.load_path,
            load_object_store=self.load_object_store.initialize_object()
            if self.load_object_store is not None else None,
            load_weights_only=self.load_weights_only,
            load_strict=self.load_strict_model_weights,
            load_chunk_size=self.load_chunk_size,
            load_progress_bar=self.load_progress_bar,
            save_folder=self.save_folder,
            save_interval=self.save_interval,
            save_compression=self.save_compression,

            # Subset parameters
            train_subset_num_batches=self.train_subset_num_batches,
            eval_subset_num_batches=self.eval_subset_num_batches,

            # DeepSpeed
            deepspeed_config=self.deepspeed,
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
