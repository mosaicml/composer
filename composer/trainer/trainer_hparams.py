# Copyright 2021 MosaicML. All Rights Reserved.

"""
Example usage and definition of hparams
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import yahp as hp

import composer
import composer.datasets as datasets
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import (BenchmarkerHparams, CallbackHparams, GradMonitorHparams, LRMonitorHparams,
                                MemoryMonitorHparams, SpeedMonitorHparams, TorchProfilerHparams)
from composer.core.types import Precision
from composer.datasets import DataloaderHparams
from composer.loggers import (BaseLoggerBackendHparams, FileLoggerBackendHparams, TQDMLoggerBackendHparams,
                              WandBLoggerBackendHparams)
from composer.models import (CIFARResNetHparams, EfficientNetB0Hparams, GPT2Hparams, MnistClassifierHparams,
                             ModelHparams, ResNet18Hparams, ResNet50Hparams, ResNet101Hparams, UnetHparams)
from composer.optim import (AdamHparams, AdamWHparams, DecoupledAdamWHparams, DecoupledSGDWHparams, OptimizerHparams,
                            RAdamHparams, RMSPropHparams, SchedulerHparams, SGDHparams, scheduler)
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import ddp

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
    "cosine_decay": scheduler.CosineAnnealingLRHparams,
    "cosine_warmrestart": scheduler.CosineAnnealingWarmRestartsHparams,
    "warmup": scheduler.WarmUpLRHparams,
    "constant": scheduler.ConstantLRHparams,
}

model_registry = {
    "unet": UnetHparams,
    "efficientnetb0": EfficientNetB0Hparams,
    "resnet56_cifar10": CIFARResNetHparams,
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
    "synthetic": datasets.SyntheticDatasetHparams,
    "mnist": datasets.MNISTDatasetHparams,
    "lm": datasets.LMDatasetHparams,
}

algorithms_registry = get_algorithm_registry()

callback_registry = {
    "torch_profiler": TorchProfilerHparams,
    "speed_monitor": SpeedMonitorHparams,
    "benchmarker": BenchmarkerHparams,
    "lr_monitor": LRMonitorHparams,
    "grad_monitor": GradMonitorHparams,
    "memory_monitor": MemoryMonitorHparams,
}

logger_registry = {
    "file": FileLoggerBackendHparams,
    "wandb": WandBLoggerBackendHparams,
    "tqdm": TQDMLoggerBackendHparams,
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

    total_batch_size: int = hp.required(
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
    ddp_sync_strategy: Optional[ddp.DDPSyncStrategy] = hp.optional(
        doc="The strategy for synchronizing DDP. Default value ``None`` causes the "
        "trainer to auto-select a value depending on what algorithms are used.",
        default=None)
    ddp_timeout: float = hp.optional(doc="Timeout, in seconds, for initializing the DDP process group.", default=5.0)

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

    checkpoint_filepath: Optional[str] = hp.optional(doc="Path to an existing checkpoint file to load from.",
                                                     default=None)

    checkpoint_interval_unit: Optional[str] = hp.optional(
        doc=
        "Unit for the checkpoint save interval -- should be 'ep' for epochs; 'ba' for batches, or None to disable checkpointing",
        default=None)
    checkpoint_interval: int = hp.optional(doc="Interval for checkpointing.", default=1)
    checkpoint_folder: str = hp.optional(
        doc="Folder in which to save checkpoint files. Relative to the run directory, if set."
        "Defaults to `checkpoints`.",
        default="checkpoints")
    deterministic_mode: bool = hp.optional(doc="Run the model deterministically. Experimental. Performance"
                                           "degradations expected. Certain Torch modules may not have"
                                           "deterministic implementations, which will result in a crash.",
                                           default=False)

    compute_training_metrics: bool = hp.optional(doc="Log validation metrics on training data", default=False)
    log_level: str = hp.optional(doc="Python loglevel to use composer", default="INFO")

    def validate(self):
        super().validate()

        world_size = ddp.get_world_size()

        if self.total_batch_size % world_size != 0:
            raise ValueError(
                f"batch size ({self.total_batch_size}) not divisible by the total number of processes ({world_size})")

        if self.eval_batch_size % world_size != 0:
            raise ValueError(
                f"eval batch size ({self.eval_batch_size}) not divisible by the total number of processes ({world_size}) "
            )

    def initialize_object(self) -> Trainer:
        from composer.trainer.trainer import Trainer
        return Trainer.create_from_hparams(hparams=self)

    def set_datadir(self, datadir: str) -> None:
        """Override the ``datadir`` property in the :attr:`train_dataset` and :attr:`val_dataset`.

        Args:
            datadir (str): The datadir
        
        Raises
            AttributeError: Raised if either :attr:`train_dataset` or :attr:`val_dataset` do not
            have a ``datadir`` property.
        """
        if not hasattr(self.train_dataset, 'datadir') or not hasattr(self.val_dataset, 'datadir'):
            raise AttributeError('Both the train and val dataset hparams must have the datadir attribute.')
        setattr(self.train_dataset, 'datadir', datadir)
        setattr(self.val_dataset, 'datadir', datadir)

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
