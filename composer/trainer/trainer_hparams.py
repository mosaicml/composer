# Copyright 2021 MosaicML. All Rights Reserved.

"""
Example usage and definition of hparams
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import yahp as hp

import composer
from composer import datasets
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import (BenchmarkerHparams, CallbackHparams, GradMonitorHparams, LRMonitorHparams,
                                MemoryMonitorHparams, RunDirectoryUploaderHparams, SpeedMonitorHparams,
                                TorchProfilerHparams)
from composer.core.types import Precision
from composer.datasets import DataloaderHparams
from composer.loggers import (BaseLoggerBackendHparams, FileLoggerBackendHparams, MosaicMLLoggerBackendHparams,
                              TQDMLoggerBackendHparams, WandBLoggerBackendHparams)
from composer.models import (CIFARResNet9Hparams, CIFARResNetHparams, EfficientNetB0Hparams, GPT2Hparams,
                             MnistClassifierHparams, ModelHparams, ResNet18Hparams, ResNet50Hparams, ResNet101Hparams,
                             UnetHparams)
from composer.optim import (AdamHparams, AdamWHparams, DecoupledAdamWHparams, DecoupledSGDWHparams, OptimizerHparams,
                            RAdamHparams, RMSPropHparams, SchedulerHparams, SGDHparams, scheduler)
from composer.trainer.checkpoint_hparams import CheckpointLoaderHparams, CheckpointSaverHparams
from composer.trainer.ddp import DDPSyncStrategy
from composer.trainer.deepspeed import DeepSpeedHparams
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist

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
    "warmup": scheduler.WarmUpLRHparams,
    "constant": scheduler.ConstantLRHparams,
}

model_registry = {
    "unet": UnetHparams,
    "efficientnetb0": EfficientNetB0Hparams,
    "resnet56_cifar10": CIFARResNetHparams,
    "resnet9_cifar10": CIFARResNet9Hparams,
    "resnet101": ResNet101Hparams,
    "resnet50": ResNet50Hparams,
    "resnet18": ResNet18Hparams,
    "mnist_classifier": MnistClassifierHparams,
    "gpt2": GPT2Hparams,
}

dataset_registry = {
    "brats": datasets.BratsDatasetHparams,
    "imagenet": datasets.ImagenetDatasetHparams,
    "cifar10": datasets.CIFAR10DatasetHparams,
    "mnist": datasets.MNISTDatasetHparams,
    "lm": datasets.LMDatasetHparams,
    "streaming_lm": datasets.StreamingLMDatasetHparams,
}

algorithms_registry = get_algorithm_registry()

callback_registry = {
    "torch_profiler": TorchProfilerHparams,
    "speed_monitor": SpeedMonitorHparams,
    "benchmarker": BenchmarkerHparams,
    "lr_monitor": LRMonitorHparams,
    "grad_monitor": GradMonitorHparams,
    "memory_monitor": MemoryMonitorHparams,
    "run_directory_uploader": RunDirectoryUploaderHparams,
}

logger_registry = {
    "file": FileLoggerBackendHparams,
    "wandb": WandBLoggerBackendHparams,
    "tqdm": TQDMLoggerBackendHparams,
    "mosaicml": MosaicMLLoggerBackendHparams,
}

device_registry = {
    "gpu": GPUDeviceHparams,
    "cpu": CPUDeviceHparams,
}


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
    }

    device: DeviceHparams = hp.required(doc="Device Parameters")
    train_dataset: datasets.DatasetHparams = hp.required(doc="Training dataset hparams")
    val_dataset: datasets.DatasetHparams = hp.required(doc="Validation dataset hparams")

    optimizer: OptimizerHparams = hp.required(doc="Optimizer to use")

    model: ModelHparams = hp.required(doc="model")
    loggers: List[BaseLoggerBackendHparams] = hp.required(doc="loggers to use")

    max_epochs: int = hp.required(
        doc="training time in epochs and/or batches (e.g., 90ep5ba)",
        template_default=10,
    )

    train_batch_size: int = hp.required(
        doc="batch size for each optimization step, across all devices and gradient accumulations.",
        template_default=2048,
    )

    eval_batch_size: int = hp.required(
        doc="batch size to use for each evaluation step",
        template_default=2048,
    )

    dataloader: DataloaderHparams = hp.required(doc="dataloader hparams")

    grad_accum: int = hp.required(
        template_default=1,
        doc=
        "Determines the number of microbatches to split a per-gpu batch into, used to compensate for low-memory-capacity devices."
    )
    precision: Precision = hp.required(doc="Precision to use for training", template_default=Precision.AMP)

    dist_timeout: float = hp.optional(doc="Timeout, in seconds, for initializing the dsitributed process group.",
                                      default=15.0)
    ddp_sync_strategy: Optional[DDPSyncStrategy] = hp.optional(
        doc="The strategy for synchronizing DDP. Default value ``None`` causes the "
        "trainer to auto-select a value depending on what algorithms are used.",
        default=None)

    deepspeed: Optional[DeepSpeedHparams] = hp.optional(doc="Configuration for DeepSpeed.", default=None)

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
    callbacks: List[CallbackHparams] = hp.optional(doc="Callback hparams", default_factory=list)

    load_checkpoint: Optional[CheckpointLoaderHparams] = hp.optional(doc="Checkpoint loading hparams", default=None)
    save_checkpoint: Optional[CheckpointSaverHparams] = hp.optional(doc="Checkpointing hparams", default=None)

    train_subset_num_batches: Optional[int] = hp.optional(textwrap.dedent("""If specified,
        finish every epoch early after training on this many batches."""),
                                                          default=None)
    eval_subset_num_batches: Optional[int] = hp.optional(textwrap.dedent("""If specified,
        stop each evaluation after this many batches."""),
                                                         default=None)

    deterministic_mode: bool = hp.optional(doc="Run the model deterministically. Experimental. Performance"
                                           "degradations expected. Certain Torch modules may not have"
                                           "deterministic implementations, which will result in a crash.",
                                           default=False)

    compute_training_metrics: bool = hp.optional(doc="Log validation metrics on training data", default=False)
    log_level: str = hp.optional(doc="Python loglevel to use composer", default="WARNING")
    datadir: Optional[str] = hp.optional(doc=textwrap.dedent("""
        Datadir to apply for both the training and validation datasets. If specified,
        it will override train_dataset.datadir and val_dataset.datadir"""),
                                         default=None)

    def validate(self):
        super().validate()

        if self.deepspeed is not None:

            if self.precision == Precision.FP16:
                raise ValueError("FP16 precision is only supported when training with DeepSpeed.")

            if isinstance(self.device, CPUDeviceHparams):
                raise ValueError("Training on CPUs is not supported with DeepSpeed.")

            if self.deterministic_mode and self.deepspeed.zero_stage > 0:
                raise ValueError("Deepspeed with zero stage > 0 is not compatible with deterministic mode")

        world_size = dist.get_world_size()

        if self.train_batch_size % world_size != 0:
            raise ValueError(
                f"Batch size ({self.train_batch_size}) not divisible by the total number of processes ({world_size}).")

        if self.eval_batch_size % world_size != 0:
            raise ValueError(
                f"Eval batch size ({self.eval_batch_size}) not divisible by the total number of processes ({world_size})."
            )

    def initialize_object(self) -> Trainer:
        from composer.trainer.trainer import Trainer
        return Trainer.create_from_hparams(hparams=self)

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
