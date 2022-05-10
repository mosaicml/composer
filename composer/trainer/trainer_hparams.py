# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`~yahp.hparams.Hparams` used to construct the :class:`~composer.trainer.trainer.Trainer`."""

from __future__ import annotations

import datetime
import logging
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import torch
import yahp as hp

import composer
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import (CallbackHparams, GradMonitorHparams, LRMonitorHparams, MemoryMonitorHparams,
                                MLPerfCallbackHparams, SpeedMonitorHparams)
from composer.core import Precision
from composer.core.types import JSON
from composer.datasets import DataLoaderHparams, DatasetHparams
from composer.datasets.dataset_registry import get_dataset_registry
from composer.datasets.evaluator import EvaluatorHparams
from composer.loggers import LoggerDestinationHparams, logger_registry
from composer.loggers.logger import LogLevel
from composer.models import (BERTForClassificationHparams, BERTHparams, DeepLabV3Hparams, EfficientNetB0Hparams,
                             GPT2Hparams, MnistClassifierHparams, ModelHparams, ResNetCIFARHparams, ResNetHparams,
                             SSDHparams, TimmHparams, UnetHparams, ViTSmallPatch16Hparams)
from composer.optim import (AdamHparams, AdamWHparams, ConstantSchedulerHparams, CosineAnnealingSchedulerHparams,
                            CosineAnnealingWarmRestartsSchedulerHparams, CosineAnnealingWithWarmupSchedulerHparams,
                            DecoupledAdamWHparams, DecoupledSGDWHparams, ExponentialSchedulerHparams,
                            LinearSchedulerHparams, LinearWithWarmupSchedulerHparams, MultiStepSchedulerHparams,
                            MultiStepWithWarmupSchedulerHparams, OptimizerHparams, PolynomialSchedulerHparams,
                            RAdamHparams, RMSpropHparams, SchedulerHparams, SGDHparams, StepSchedulerHparams)
from composer.profiler.profiler_hparams import ProfilerHparams
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
    "mlperf": MLPerfCallbackHparams,
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

        datadir (str, optional): Datadir to apply for both the training and validation datasets. If specified,
            it will override both ``train_dataset.datadir`` and ``val_dataset.datadir``. (default: ``None``)
        dataloader (DataLoaderHparams): Hparams used for constructing the dataloader which will be used
            for loading the train dataset and (if provided) the validation dataset.

        train_dataset (DatasetHparams): Hparams used to construct the dataset used for training.

            .. seealso:: :mod:`composer.datasets` for datasets built into Composer.
        train_dataloader_label (str): See :class:`.Trainer`.
        train_batch_size (int): The optimization batch size to use for training. This is the total batch
            size that is used to produce a gradient for the optimizer update step.
        train_subset_num_batches (int, optional): See :class:`.Trainer`.
        compute_training_metrics (bool, optional): See :class:`.Trainer`.

        max_duration (str): The maximum duration to train as a str (e.g. ``1ep``, or ``10ba``).
            Will be converted to a :class:`~composer.core.Time` object.

            .. seealso:: :class:`~composer.core.Time` for more details on time construction.

        algorithms (List[AlgorithmHparams], optional): The algorithms to use during training. (default: ``[]``)

            .. seealso:: :mod:`composer.algorithms` for the different algorithms built into Composer.

        optimizers (OptimizerHparams, optional): The hparams for constructing the optimizer. (default: ``None``)

            .. seealso:: :class:`.Trainer` for the default optimizer behavior when ``None`` is provided.

            .. seealso:: :mod:`composer.optim` for the different optimizers built into Composer.
        schedulers (List[SchedulerHparams], optional): The learning rate schedulers. (default: ``[]``).

            .. seealso:: :class:`.Trainer` for the default scheduler behavior when ``[]`` is provided.

            .. seealso:: :mod:`composer.optim.scheduler` for the different schedulers built into Composer.
        scale_schedule_ratio (float, optional): See :class:`.Trainer`.
        step_schedulers_every_batch (bool, optional): See :class:`.Trainer`.

        val_dataset (DatasetHparams, optional): Hparams for constructing the dataset used for evaluation.
            (default: ``None``)

            .. seealso:: :mod:`composer.datasets` for datasets built into Composer.
        evaluators (List[EvaluatorHparams], optional): Hparams for constructing evaluators to be used during the
            eval loop. Evaluators should be used when evaluating one or more specific metrics across one
            or more datasets. (default: ``None``)

            .. seealso:: :class:`~composer.core.evaluator.Evaluator` for more details on evaluators.
        eval_batch_size (int, optional): The batch size to use for evaluation. Must be provided if one of
            ``val_dataset`` or ``evaluators`` is set. (default: ``None``)
        eval_interval (str, optional): See :class:`.Trainer`.
        eval_subset_num_batches (int, optional): See :class:`.Trainer`.

        callbacks (List[CallbackHparams], optional): Hparams to construct the callbacks to
            run during training. (default: ``[]``)

            .. seealso:: :mod:`composer.callbacks` for the different callbacks built into Composer.

        loggers (List[LoggerDestinationHparams], optional): Hparams for constructing the destinations
            to log to. (default: ``[]``)

            .. seealso:: :mod:`composer.loggers` for the different loggers built into Composer.
        run_name (str, optional): See :class:`.Trainer`.
        progress_bar (bool, optional): See :class:`.Trainer`.
        log_to_console (bool, optional): See :class:`.Trainer`.
        console_log_level (bool, optional): See :class:`.Trainer`.
        console_stream (bool, optional): See :class:`.Trainer`.
        python_log_level (str): The Python log level to use for log statements in the :mod:`composer`
            module. (default: ``INFO``)

            .. seealso:: The :mod:`logging` module in Python.

        load_path (str, optional): See :class:`.Trainer`.
        load_object_store (ObjectStore, optional): See :class:`.Trainer`. Both ``load_logger_destination`` and
            ``load_object_store`` should not be provided since there can only be one location to load from.
        load_logger_destination (LoggerDestination, optional): Used to specify a ``LoggerDestination`` for
            ``load_object_store`` in :class:`.Trainer` as Hparams doesn't support a Union type for those objects. Both
            ``load_logger_destination`` and ``load_object_store`` should not be provided since there can only be one location
            to load from.
        load_weights_only (bool, optional): See :class:`.Trainer`.
        load_strict_model_weights (bool, optional): See :class:`.Trainer`.
        load_chunk_size (int, optional): See :class:`.Trainer`.
        load_progress_bar (bool, optional): See :class:`.Trainer`.

        save_folder (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_filename (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_artifact_name (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_latest_filename (str, optional): See
            :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_latest_artifact_name (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_overwrite (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_weights_only (bool, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.
        save_interval (str, optional): See
            :class:`~composer.callbacks.callback_hparams.CheckpointSaverHparams`.
        save_num_checkpoints_to_keep (int, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.

        deepspeed (Dict[str, JSON], optional): If set to a dict will be used for as the DeepSpeed
            config for training  (see :class:`.Trainer` for more details). If ``None`` (the default), DeepSpeed will not
            be used.

        device (DeviceHparams, optional): Hparams for constructing the device used for training.
            (default: ``None``)
        precision (Precision, optional): See :class:`.Trainer`.
        grad_accum (int, optional): See :class:`.Trainer`.

        seed (int, optional): See :class:`.Trainer`.
        deterministic_mode (bool, optional): See :class:`.Trainer`.

        dist_timeout (float, optional): See :class:`.Trainer`.
        ddp_sync_strategy (DDPSyncStrategy, optional): See :class:`.Trainer`.

        grad_clip_norm (float, optional): See :class:`.Trainer`.

        profiler (ProfilerHparams, optional): Profiler hyperparameters.
    """

    hparams_registry = {  # type: ignore
        "algorithms": algorithms_registry,
        "optimizer": optimizer_registry,
        "schedulers": scheduler_registry,
        "loggers": logger_registry,
        "load_logger_destination": logger_registry,
        "model": model_registry,
        "train_dataset": dataset_registry,
        "val_dataset": dataset_registry,
        "callbacks": callback_registry,
        "device": device_registry,
        "evaluators": evaluator_registry,
    }

    model: ModelHparams = hp.required(doc="model")

    # Shared data
    datadir: Optional[str] = hp.optional(
        doc=(("Datadir to apply for both the training and validation datasets. If specified, "
              "it will override train_dataset.datadir and val_dataset.datadir.")),
        default=None,
    )
    dataloader: DataLoaderHparams = hp.optional(doc="dataloader hparams", default=DataLoaderHparams())

    # Train Data
    train_dataset: Optional[DatasetHparams] = hp.optional(doc="Training dataset hparams", default=None)
    train_dataloader_label: str = hp.optional(doc="Train dataset label", default="train")
    train_batch_size: Optional[int] = hp.optional(
        doc="batch size for each optimization step, across all devices and gradient accumulations.",
        default=None,
    )
    train_subset_num_batches: int = hp.optional(
        doc="If specified, finish every epoch early after training on this many batches.",
        default=-1,
    )
    compute_training_metrics: bool = hp.optional(doc="Log validation metrics on training data", default=False)

    # Stopping Conditions
    max_duration: Optional[Union[str, int]] = hp.optional(
        doc="Time string for the maximum training duration (e.g., 90ep)",
        default=None,
    )

    # Algorithms
    algorithms: List[AlgorithmHparams] = hp.optional(doc="Algorithms", default_factory=list)

    # Optimizer and Scheduler
    optimizer: Optional[OptimizerHparams] = hp.optional(doc="Optimizer to use", default=None)
    schedulers: List[SchedulerHparams] = hp.optional(doc="Schedulers", default_factory=list)
    scale_schedule_ratio: float = hp.optional(
        doc="Ratio by which to scale the training duration and learning rate schedules.",
        default=1.0,
    )
    step_schedulers_every_batch: bool = hp.optional(
        doc="Whether schedulers will update after every optimizer step (True), or every epoch (False).",
        default=True,
    )

    # Evaluation
    val_dataset: Optional[DatasetHparams] = hp.optional(doc="Validation dataset hparams", default=None)
    evaluators: Optional[List[EvaluatorHparams]] = hp.optional(doc="Evaluators", default=None)
    eval_batch_size: Optional[int] = hp.optional(doc="batch size to use for each evaluation step", default=None)
    eval_interval: str = hp.optional(
        doc="Time string for the evaluation interval. Defaults to 1ep (every epoch)",
        default="1ep",
    )
    eval_subset_num_batches: int = hp.optional(
        doc="If specified, stop each evaluation after this many batches.",
        default=-1,
    )

    # Callbacks
    callbacks: List[CallbackHparams] = hp.optional(doc="Callback hparams", default_factory=list)

    # Logging
    loggers: List[LoggerDestinationHparams] = hp.optional(doc="loggers to use", default_factory=list)
    run_name: Optional[str] = hp.optional("Experiment name", default=None)
    progress_bar: bool = hp.optional("Whether to show a progress bar.", default=True)
    log_to_console: Optional[bool] = hp.optional("Whether to print log statements to the console.", default=None)
    console_log_level: LogLevel = hp.optional("The maximum log level for console logging.", default=LogLevel.EPOCH)
    console_stream: str = hp.optional(
        doc="The stream at which to write the progress bar and log statements.",
        default="stderr",
    )
    python_log_level: str = hp.optional(doc="Python loglevel to use composer", default="INFO")

    # Load Checkpoint
    load_path: Optional[str] = hp.optional(
        doc=((
            "If specified, the path to an existing checkpoint file "
            "(if the checkpoint is on the local disk) or the object name for the checkpoint "
            "(if the checkpoint is in a cloud bucket). Set to None (the default) to skip loading from a checkpoint.")),
        default=None,
    )
    load_object_store: Optional[ObjectStoreHparams] = hp.optional(
        doc=(("If the checkpoint is in an object store (i.e. AWS S3 or Google Cloud Storage), the parameters for "
              "connecting to the cloud provider object store. Otherwise, if the checkpoint is a local filepath, "
              "leave blank. This parameter has no effect if `load_path` is not specified.")),
        default=None)
    load_logger_destination: Optional[LoggerDestinationHparams] = hp.optional(
        ("Alternative argument to `load_object_store` to support loading from a logger destination. This parameter "
         "has no effect if `load_path` is not specified or `load_object_store` is specified, which will be "
         "used instead of this."),
        default=None)
    load_weights_only: bool = hp.optional(
        doc=(("Whether to only load the weights from the model. "
              "This parameter has no effect if `load_path`is not specified.")),
        default=False,
    )
    load_strict_model_weights: bool = hp.optional(
        doc=(("Ensure that the set of checkpoint weights in the checkpoint and model must exactly match. "
              "This parameter has no effect if `load_path` is not specified.")),
        default=False,
    )

    load_chunk_size: int = hp.optional(
        doc=(("Chunk size (in bytes) to use when downloading checkpoints. "
              "This parameter has no effect if `load_path` is not specified or it is a local file path.")),
        default=1_048_576,
    )
    load_progress_bar: bool = hp.optional(
        doc=(("Whether to show a progress bar when downloading checkpoints. "
              "This parameter has no effect if `load_path` is not specified or it is a local file path.")),
        default=True,
    )

    # Save Checkpoint
    save_folder: Optional[str] = hp.optional(doc="Checkpoint folder format string.", default=None)
    save_filename: str = hp.optional(doc="Checkpoint name format string.", default="ep{epoch}-ba{batch}-rank{rank}")
    save_artifact_name: str = hp.optional(
        doc="Checkpoint artifact name format",
        default="{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}",
    )
    save_latest_filename: str = hp.optional(
        doc="Latest checkpoint symlink format string.",
        default="latest-rank{rank}",
    )
    save_latest_artifact_name: str = hp.optional(
        doc="Checkpoint symlink artifact name format",
        default="{run_name}/checkpoints/latest-rank{rank}",
    )
    save_overwrite: bool = hp.optional("Whether to override existing checkpoints.", default=False)
    save_weights_only: bool = hp.optional("Whether to save only checkpoint weights", default=False)
    save_interval: str = hp.optional(
        doc=(("Checkpoint interval or path to a `(State, Event) -> bool` function returning whether a checkpoint "
              "should be saved.")),
        default="1ep",
    )
    save_num_checkpoints_to_keep: int = hp.optional(
        doc="Number of checkpoints to persist locally. Set to -1 to never delete checkpoints.",
        default=-1,
    )

    # DeepSpeed
    deepspeed: Optional[Dict[str, JSON]] = hp.optional(doc="Configuration for DeepSpeed.", default=None)

    # System/Numerics
    device: Optional[DeviceHparams] = hp.optional(doc="Device Parameters", default=None)
    precision: Optional[Precision] = hp.optional(doc="Precision to use for training", default=None)
    grad_accum: Union[int, str] = hp.optional(
        doc=(("Determines the number of microbatches to split a per-gpu batch into, "
              "used to compensate for low-memory-capacity devices. If set to auto, "
              "dynamically increases grad_accum if microbatch size is too large for "
              "GPU. Defaults to ``1``")),
        default=1,
    )

    # Reproducibility
    seed: Optional[int] = hp.optional(default=None, doc="random seed to set")
    deterministic_mode: bool = hp.optional(
        doc=(("Run the model deterministically. Experimental. Performance "
              "degradations expected. Certain Torch modules may not have "
              "deterministic implementations, which will result in a crash.")),
        default=False,
    )

    # Distributed
    dist_timeout: float = hp.optional(
        doc="Timeout, in seconds, for initializing the distributed process group.",
        default=300.0,
    )
    ddp_sync_strategy: Optional[DDPSyncStrategy] = hp.optional(
        doc=(("The strategy for synchronizing DDP. Default value ``None`` causes the "
              "trainer to auto-select a value depending on what algorithms are used.")),
        default=None,
    )

    # Grad Clip Norm
    grad_clip_norm: float = hp.optional(
        default=-1.0,
        doc='The norm to clip gradient magnitudes to. Default: -1 (no clip)',
    )

    # Profiling
    profiler: Optional[ProfilerHparams] = hp.optional(doc="Profiler parameters", default=None)

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

        world_size = dist.get_world_size()

        if self.train_batch_size is not None and self.train_batch_size % world_size != 0:
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

        # Set the Python LogLevel for Composer
        import composer
        logging.getLogger(composer.__name__).setLevel(self.python_log_level)

        # Device
        device_hparams = self.device
        if device_hparams is None:
            device_hparams = GPUDeviceHparams() if torch.cuda.is_available() else CPUDeviceHparams()
        device = device_hparams.initialize_object()

        # Distributed
        # Initialized here so it is available within dataloaders
        if dist.get_world_size() > 1:
            dist.initialize_dist(device.dist_backend, datetime.timedelta(seconds=self.dist_timeout))

        # Reproducibility
        seed = self.seed if self.seed else reproducibility.get_random_seed()
        # need to set seed before model initialization for determinism
        # don't need to set different seeds per process since only the rank 0 initialization is used
        # Algorithms should not use the `seed` on `__init__` but rather on `Event.INIT`, which occurs
        # after the seed was properly distributed across ranks to ensure checkpoint compatibility
        reproducibility.seed_all(seed)

        # The model
        model = self.model.initialize_object()

        # Loggers, Callbacks, and Algorithms
        loggers = [x.initialize_object() for x in self.loggers]
        callbacks = [x.initialize_object() for x in self.callbacks]
        algorithms = [x.initialize_object() for x in self.algorithms]

        # Shared data configuration
        if self.datadir is not None:
            if self.train_dataset is not None:
                self.train_dataset.datadir = self.datadir
            if self.val_dataset is not None:
                self.val_dataset.datadir = self.datadir

        # Train DataLoader
        train_dataloader = None
        if self.train_dataset is not None:
            if self.train_batch_size is None:
                raise ValueError("The train batch size must be specified if the train_dataset is specified")

            train_device_batch_size = self.train_batch_size // dist.get_world_size()
            if self.train_dataset.shuffle and self.train_subset_num_batches is not None:
                warnings.warn(
                    ("SubsetNumBatchesWarning: When specifying train_subset_num_batches, "
                     f"(set to {self.train_subset_num_batches}), train_datset.shuffle should be set to False. "
                     "Otherwise, each training epoch may load a different subset of samples."))
            train_dataloader = self.train_dataset.initialize_object(train_device_batch_size, self.dataloader)

        # Evaluation
        eval_device_batch_size = (self.eval_batch_size or 0) // dist.get_world_size()
        eval_dataloader = None
        if self.val_dataset is not None:
            if self.val_dataset.shuffle and self.eval_subset_num_batches is not None:
                warnings.warn(("SubsetNumBatchesWarning: When specifying eval_subset_num_batches, "
                               f"(set to {self.eval_subset_num_batches}), val_dataset.shuffle should be "
                               "set to False. Otherwise, each evaluation epoch may load a different "
                               "subset of samples."))
            eval_dataloader = self.val_dataset.initialize_object(eval_device_batch_size, self.dataloader)
        if self.evaluators is not None and len(self.evaluators) > 0:
            eval_dataloader = [
                evaluator.initialize_object(model, eval_device_batch_size, self.dataloader)
                for evaluator in self.evaluators
            ]
            for evaluator in self.evaluators:
                if evaluator.eval_dataset.shuffle and self.eval_subset_num_batches is not None:
                    warnings.warn(("SubsetNumBatchesWarning: When specifying eval_subset_num_batches, "
                                   f"(set to {self.eval_subset_num_batches}), evaluator.dataloader.shuffle "
                                   f"(for Evaluator: '{evaluator.label}') should be set to False. Otherwise, "
                                   "each evaluation epoch may load a different subset of samples."))

        # Optimizers and Schedulers
        optimizer = self.optimizer.initialize_object(model.parameters()) if self.optimizer is not None else None
        schedulers = [scheduler.initialize_object() for scheduler in self.schedulers]

        load_object_store = None
        if self.load_object_store is not None and self.load_logger_destination is not None:
            raise ValueError(
                "load_object_store and load_logger_destination cannot both be non-None. Please provide only one location to load from."
            )
        elif self.load_object_store is not None:
            load_object_store = self.load_object_store.initialize_object()
        elif self.load_logger_destination is not None:
            load_object_store = self.load_logger_destination.initialize_object()

        trainer = Trainer(
            # Model
            model=model,

            # Train Data
            train_dataloader=train_dataloader,
            train_dataloader_label=self.train_dataloader_label,
            compute_training_metrics=self.compute_training_metrics,
            train_subset_num_batches=self.train_subset_num_batches,

            # Stopping Condition
            max_duration=self.max_duration,

            # Algorithms
            algorithms=algorithms,

            # Optimizers and Schedulers
            optimizers=optimizer,
            schedulers=schedulers,
            scale_schedule_ratio=self.scale_schedule_ratio,
            step_schedulers_every_batch=self.step_schedulers_every_batch,

            # Evaluation
            eval_dataloader=eval_dataloader,
            eval_interval=self.eval_interval,
            eval_subset_num_batches=self.eval_subset_num_batches,

            # Callbacks
            callbacks=callbacks,

            # Logging
            loggers=loggers,
            run_name=self.run_name,
            progress_bar=self.progress_bar,
            log_to_console=self.log_to_console,
            console_log_level=self.console_log_level,
            console_stream=self.console_stream,

            # Checkpoint Loading
            load_path=self.load_path,
            load_object_store=load_object_store,
            load_weights_only=self.load_weights_only,
            load_strict=self.load_strict_model_weights,
            load_chunk_size=self.load_chunk_size,
            load_progress_bar=self.load_progress_bar,

            # Checkpoint Saving
            save_folder=self.save_folder,
            save_overwrite=self.save_overwrite,
            save_filename=self.save_filename,
            save_latest_filename=self.save_latest_filename,
            save_artifact_name=self.save_artifact_name,
            save_interval=self.save_interval,
            save_weights_only=self.save_weights_only,
            save_num_checkpoints_to_keep=self.save_num_checkpoints_to_keep,

            # DeepSpeed
            deepspeed_config=self.deepspeed,

            # System/Numerics
            device=device,
            precision=self.precision,
            grad_accum=self.grad_accum,

            # Reproducibility
            seed=seed,
            deterministic_mode=self.deterministic_mode,

            # Distributed
            dist_timeout=self.dist_timeout,
            ddp_sync_strategy=self.ddp_sync_strategy,

            # Grad Clip Norm
            grad_clip_norm=self.grad_clip_norm,

            # Profiler
            profiler=None if self.profiler is None else self.profiler.initialize_object(),
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
