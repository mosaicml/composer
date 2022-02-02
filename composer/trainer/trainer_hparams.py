# Copyright 2021 MosaicML. All Rights Reserved.

"""
Example usage and definition of hparams
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, cast

import yahp as hp

import composer
from composer import datasets
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import (BenchmarkerHparams, CallbackHparams, GradMonitorHparams, LRMonitorHparams,
                                MemoryMonitorHparams, RunDirectoryUploaderHparams, SpeedMonitorHparams)
from composer.core.types import JSON, Precision
from composer.datasets import DataloaderHparams
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.evaluator import EvaluatorHparams
from composer.loggers import (BaseLoggerBackendHparams, FileLoggerBackendHparams, MosaicMLLoggerBackendHparams,
                              TQDMLoggerBackendHparams, WandBLoggerBackendHparams)
from composer.models import (BERTForClassificationHparams, BERTHparams, CIFARResNet9Hparams, CIFARResNetHparams,
                             DeepLabV3Hparams, EfficientNetB0Hparams, GPT2Hparams, MnistClassifierHparams, ModelHparams,
                             ResNetHparams, TimmHparams, UnetHparams)
from composer.models.resnet20_cifar10.resnet20_cifar10_hparams import CIFARResNet20Hparams
from composer.optim import (AdamHparams, AdamWHparams, DecoupledAdamWHparams, DecoupledSGDWHparams, OptimizerHparams,
                            RAdamHparams, RMSPropHparams, SchedulerHparams, SGDHparams, scheduler)
from composer.profiler import ProfilerHparams
from composer.trainer.ddp import DDPSyncStrategy
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist
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
    "warmup": scheduler.WarmUpLRHparams,
    "constant": scheduler.ConstantLRHparams,
    "polynomial": scheduler.PolynomialLRHparams,
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

    optimizer: OptimizerHparams = hp.required(doc="Optimizer to use")

    model: ModelHparams = hp.required(doc="model")
    loggers: List[BaseLoggerBackendHparams] = hp.required(doc="loggers to use")

    max_duration: str = hp.required(
        doc="Time string for the maximum training duration (e.g., 90ep)",
        template_default="10ep",
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

    profiler: Optional[ProfilerHparams] = hp.optional(doc="Profiler hparams", default=None)

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
