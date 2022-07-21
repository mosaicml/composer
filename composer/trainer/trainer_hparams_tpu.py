# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`~yahp.hparams.Hparams` used to construct the :class:`~composer.trainer.trainer.Trainer`."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
import yahp as hp
from torchmetrics import Metric, MetricCollection

import composer
#from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.algorithms.algorithm_hparams_registry import algorithm_registry
from composer.callbacks.callback_hparams_registry import callback_registry
from composer.core import Algorithm, Callback, DataSpec, Evaluator, Event, Precision, State, Time
from composer.core.types import JSON, PyTorchScheduler
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.dataset_hparams_registry import dataset_registry
from composer.datasets.evaluator_hparams import EvaluatorHparams
from composer.loggers import LoggerDestination, LogLevel
from composer.loggers.logger_hparams_registry import logger_registry
from composer.models import (BERTForClassificationHparams, BERTHparams, DeepLabV3Hparams, EfficientNetB0Hparams,
                             GPT2Hparams, MnistClassifierHparams, ModelHparams, ResNetCIFARHparams, ResNetHparams,
                             SSDHparams, TimmHparams, UnetHparams, ViTSmallPatch16Hparams)
from composer.models.base import ComposerModel
from composer.optim import ComposerScheduler
from composer.optim.optimizer_hparams_registry import OptimizerHparams, optimizer_registry
from composer.optim.scheduler_hparams_registry import scheduler_registry
from composer.profiler import Profiler
from composer.trainer.ddp import DDPSyncStrategy
from composer.trainer.devices import Device, DeviceCPU, DeviceGPU, DeviceTPU
from composer.trainer.devices.device_hparams_registry import device_registry
from composer.trainer.trainer_tpu import TrainerTPU
from composer.utils import dist, reproducibility

from composer.utils.object_store.object_store_hparams import ObjectStoreHparams, object_store_registry
import torch_xla.core.xla_model as xm

if TYPE_CHECKING:
    from typing import TypedDict
else:
    TypedDict = object  # TypedDict is not available on python 3.7


__all__ = ["TrainerTPUHparams"]#, "FitHparams", "EvalHparams", "ExperimentHparams"]

Scheduler = Union[ComposerScheduler, PyTorchScheduler]


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

evaluator_registry = {"evaluator": EvaluatorHparams}


def _initialize_dataloader(
    dataset_hparams: Optional[DatasetHparams],
    dataloader_label: str,
    batch_size: Optional[int],
    subset_num_batches: Optional[int],
    dataloader_hparams: DataLoaderHparams,
):
    """Helper method to validate dataloader arguments and initialize the dataloader."""
    dataloader = None
    if dataset_hparams is not None:
        if batch_size is None:
            raise ValueError(
                f"The batch size for {dataloader_label} must be specified if the {dataloader_label} dataset is specified"
            )

        #train_device_batch_size = batch_size // dist.get_world_size()
        import torch_xla.core.xla_model as xm
        train_device_batch_size = batch_size // xm.xrt_world_size()
        
        if dataset_hparams.shuffle and subset_num_batches is not None:
            warnings.warn(
                (f"SubsetNumBatchesWarning: When specifying `subset_num_batches` for the {dataloader_label} dataset, "
                 f"dataset_hparams.shuffle should be set to False. "
                 "Otherwise, each epoch may load a different subset of samples."))
        dataloader = dataset_hparams.initialize_object(train_device_batch_size, dataloader_hparams)
    return dataloader


def _initialize_eval_dataloader(
    model: ComposerModel,
    eval_dataset_hparams: Optional[DatasetHparams],
    evaluators: Optional[List[EvaluatorHparams]],
    eval_batch_size: Optional[int],
    eval_subset_num_batches: Optional[int],
    dataloader_hparams: DataLoaderHparams,
):
    """Helper method to evaluation arguments and initialize the eval_dataloader."""
    eval_dataloader = None
    if eval_dataset_hparams is not None and evaluators is not None:
        raise ValueError(
            "Either `eval_dataset` or `evaluators` should be specified. It is not permitted to specify both.")
    if eval_dataset_hparams is not None:
        eval_dataloader = _initialize_dataloader(
            eval_dataset_hparams,
            "eval",
            eval_batch_size,
            eval_subset_num_batches,
            dataloader_hparams,
        )
    if evaluators is not None:
        eval_device_batch_size = (eval_batch_size or 0) // dist.get_world_size()
        eval_dataloader = [
            evaluator.initialize_object(model, eval_device_batch_size, dataloader_hparams) for evaluator in evaluators
        ]
        for evaluator in evaluators:
            if evaluator.eval_dataset.shuffle and eval_subset_num_batches is not None:
                warnings.warn(("SubsetNumBatchesWarning: When specifying eval_subset_num_batches, "
                               f"(set to {eval_subset_num_batches}), evaluator.dataloader.shuffle "
                               f"(for Evaluator: '{evaluator.label}') should be set to False. Otherwise, "
                               "each evaluation epoch may load a different subset of samples."))
    return eval_dataloader


@dataclasses.dataclass
class TrainerTPUHparams(hp.Hparams):

    hparams_registry = {  # type: ignore
        'algorithms': algorithm_registry,
        'optimizers': optimizer_registry,
        'schedulers': scheduler_registry,
        'loggers': logger_registry,
        'load_logger_destination': logger_registry,
        'model': model_registry,
        'train_dataset': dataset_registry,
        'val_dataset': dataset_registry,
        'callbacks': callback_registry,
        'device': device_registry,
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
    algorithms: List[Algorithm] = hp.optional(doc="Algorithms", default_factory=list)

    # Optimizer and Scheduler
    optimizers: Optional[OptimizerHparams] = hp.optional(doc="Optimizer to use", default=None)
    schedulers: Optional[List[ComposerScheduler]] = hp.optional(doc='sched', default=None)
    
    scale_schedule_ratio: float = hp.optional(
        doc="Ratio by which to scale the training duration and learning rate schedules.",
        default=1.0,
    )
    step_schedulers_every_batch: Optional[bool] = hp.optional(
        doc="Whether schedulers will update after every optimizer step (True), or every epoch (False).",
        default=None,
    )

    # Evaluation
    val_dataset: Optional[DatasetHparams] = hp.optional(doc="Validation dataset hparams", default=None)
    evaluators: Optional[List[EvaluatorHparams]] = hp.optional(doc="Evaluators", default=None)
    eval_batch_size: Optional[int] = hp.optional(doc="batch size to use for each evaluation step", default=None)
    eval_interval: Union[int, str] = hp.optional(
        doc="Time string or integers in epochs for the evaluation interval. Defaults to 1 (every epoch)",
        default=1,
    )
    eval_subset_num_batches: int = hp.optional(
        doc="If specified, stop each evaluation after this many batches.",
        default=-1,
    )

    # Callbacks
    callbacks: List[Callback] = hp.optional(doc="Callback hparams", default_factory=list)

    # Logging
    #loggers: List[LoggerDestinationHparams] = hp.optional(doc="loggers to use", default_factory=list)
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

    # Graceful Resumption
    autoresume: bool = hp.optional(doc=(("Whether or not to enable autoresume, which allows for stopping and resuming "
                                         "training. This parameter requires ``save_folder`` and ``run_name`` to "
                                         "be specified and ``save_overwrite`` to be ``False``. ")),
                                   default=False)

    # System/Numerics
    #device: Optional[DeviceHparams] = hp.optional(doc="Device Parameters", default=None)
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

    # Grad Clip Norm
    grad_clip_norm: float = hp.optional(
        default=-1.0,
        doc='The norm to clip gradient magnitudes to. Default: -1 (no clip)',
    )

    # Profiling
    profiler: Optional[Profiler] = hp.optional(doc="Profiler parameters", default=None)

    def validate(self):
        super().validate()

        world_size = xm.xrt_world_size()

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
            raise ValueError('grad_accum must be "auto" or an int greater than or equal to 1.')

    def initialize_object(self) -> Trainer:
        self.validate()

        # Set the Python LogLevel for Composer
        import composer
        logging.getLogger(composer.__name__).setLevel(self.python_log_level)

        #device_hparams = TPUDeviceHparams()
        
        device = DeviceTPU()#device_hparams.initialize_object()

        # Distributed
        # Initialized here so it is available within dataloaders
        # TODO: check for xm
        if False:#dist.get_world_size() > 1:# and device is not "tpu":
            dist.initialize_dist(device.dist_backend, datetime.timedelta(seconds=self.dist_timeout))

        # Reproducibility
        seed = self.seed if self.seed else reproducibility.get_random_seed()
        reproducibility.seed_all(seed)

        # The model
        model = self.model.initialize_object()

        # Loggers, Callbacks, and Algorithms
        #loggers = [x.initialize_object() for x in self.loggers]
        callbacks = [x.initialize_object() for x in self.callbacks]
        algorithms = [x.initialize_object() for x in self.algorithms]

        # Train dataloader
        train_dataloader = _initialize_dataloader(self.train_dataset, self.train_dataloader_label,
                                                  self.train_batch_size, self.train_subset_num_batches, self.dataloader)

        # Evaluation
        eval_dataloader = _initialize_eval_dataloader(
            model,
            self.val_dataset,
            self.evaluators,
            self.eval_batch_size,
            self.eval_subset_num_batches,
            self.dataloader,
        )

        # Optimizers and Schedulers
        optimizer = self.optimizers.initialize_object(model.parameters()) if self.optimizers is not None else None
        schedulers = [scheduler.initialize_object() for scheduler in self.schedulers]

        load_object_store = None
        if self.load_object_store is not None and self.load_logger_destination is not None:
            raise ValueError(
                "load_object_store and load_logger_destination cannot both be non-None. Please provide only one location to load from."
            )
        
        trainer = TrainerTPU(
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
            #loggers=loggers,
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

            # Graceful Resumption
            autoresume=self.autoresume,

            # System/Numerics
            device=device,
            precision=self.precision,
            grad_accum=self.grad_accum,

            # Reproducibility
            seed=seed,
            deterministic_mode=self.deterministic_mode,

            # Distributed

            # Grad Clip Norm
            grad_clip_norm=self.grad_clip_norm,

            # Profiler
            profiler=None if self.profiler is None else self.profiler.initialize_object(),
        )

        return trainer

    @classmethod
    def load(cls, model: str) -> TrainerTPUHparams:
        model_hparams_file = os.path.join(
            os.path.dirname(composer.__file__),
            "yamls",
            "models",
            f"{model}.yaml",
        )
        trainer_hparams = TrainerTPUHparams.create(model_hparams_file, cli_args=False)
        assert isinstance(trainer_hparams, TrainerTPUHparams), "trainer hparams should return an instance of self"
        return trainer_hparams


class FitKwargs(TypedDict):
    """Typing annotation for kwargs that can be passed into :meth:`.Trainer.fit`.

    :meta private:
    """
    train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]]
    train_dataloader_label: str
    train_subset_num_batches: Optional[int]
    compute_training_metrics: Optional[bool]

    # Timing
    reset_time: bool
    duration: Optional[Union[int, str, Time[int]]]

    # Schedulers
    schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]]
    scale_schedule_ratio: float
    step_schedulers_every_batch: Optional[bool]

    # Evaluation
    eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]]
    eval_subset_num_batches: int
    eval_interval: Union[int, str, Time, Callable[[State, Event], bool]]

    # Numerics
    grad_accum: Optional[Union[int, str]]
    precision: Optional[Union[str, Precision]]

    # Grad Clipping
    grad_clip_norm: Optional[float]


@dataclasses.dataclass
class FitHparams(hp.Hparams):
    """Hyperparameters to describe a call to :meth:`.Trainer.fit`.

    Args:
        train_dataset (DatasetHparams, optional): Train dataset hyperparameters.
        train_batch_size (int, optional): The optimization batch size for the training dataset.
        train_dataloader_label (str, optional): See :meth:`.Trainer.fit`.
        train_subset_num_batches (int, optional): See :meth:`.Trainer.fit`.
        compute_training_metrics (bool, optional): See :meth:`.Trainer.fit`.
        reset_time (bool, optional): See :meth:`.Trainer.fit`.
        duration (int | str, optional): See :meth:`.Trainer.fit`.
        schedulers (List[SchedulerHparams], optional): Scheduler hyperparameters.
        scale_schedule_ratio (float, optional): See :meth:`.Trainer.fit`.
        step_schedulers_every_batch (bool, optional): See :meth:`.Trainer.fit`.
        eval_dataset (DatasetHparams, optional): Validation dataset hyperameters.
            Cannot be specified with ``evaluators``.
        evaluators (EvaluatorHparams, optional): Evaluator hyperparameters.
            Cannot be specified with ``eval_dataset``.
        eval_batch_size (int, optional): See :meth:`.Trainer.fit`.
        eval_interval (int | str, optional): See :meth:`.Trainer.fit`.
        eval_subset_num_batches (int, optional): See :meth:`.Trainer.fit`.
        precision (Precision, optional): See :meth:`.Trainer.fit`.
        grad_accum (int, optional): See :meth:`.Trainer.fit`.
        grad_clip_norm (float, optional): See :meth:`.Trainer.fit`.
    """

    hparams_registry = {
        "train_dataset": dataset_registry,
        "schedulers": scheduler_registry,
        "eval_dataset": dataset_registry,
    }
    # Train dataloader
    train_dataset: Optional[DatasetHparams] = hp.required("Train dataset")
    train_batch_size: Optional[int] = hp.optional(
        doc="batch size for each optimization step, across all devices and gradient accumulations.",
        default=None,
    )
    train_dataloader_label: str = hp.optional("Train dataloader label", default="train")
    train_subset_num_batches: Optional[int] = hp.optional("Train subset num batches", default=None)
    compute_training_metrics: Optional[bool] = hp.optional("Whether to compute training metrics", default=None)

    # Timing
    reset_time: bool = hp.optional("Whether to reset the time", default=False)
    duration: Optional[Union[int, str]] = hp.optional("Duration", default=None)

    # Schedulers
    schedulers: Optional[List[SchedulerHparams]] = hp.optional(doc="Schedulers", default=None)
    scale_schedule_ratio: float = hp.optional(
        doc="Ratio by which to scale the training duration and learning rate schedules.",
        default=1.0,
    )
    step_schedulers_every_batch: Optional[bool] = hp.optional(
        doc="Whether schedulers will update after every optimizer step (True), or every epoch (False).",
        default=None,
    )

    # Evaluation
    eval_dataset: Optional[DatasetHparams] = hp.optional(doc="Validation dataset hparams", default=None)
    evaluators: Optional[List[EvaluatorHparams]] = hp.optional(doc="Evaluators", default=None)
    eval_batch_size: Optional[int] = hp.optional(doc="batch size to use for each evaluation step", default=None)
    eval_interval: Union[int, str] = hp.optional(
        doc="Time string for the evaluation interval. Defaults to 1 (every epoch)",
        default=1,
    )
    eval_subset_num_batches: int = hp.optional(
        doc="If specified, stop each evaluation after this many batches.",
        default=-1,
    )

    # Numerics
    precision: Optional[Precision] = hp.optional(doc="Precision to use for training", default=None)
    grad_accum: Optional[Union[int, str]] = hp.optional(
        doc=(("Determines the number of microbatches to split a per-gpu batch into, "
              "used to compensate for low-memory-capacity devices. If set to auto, "
              "dynamically increases grad_accum if microbatch size is too large for "
              "GPU. Defaults to ``None``")),
        default=None,
    )

    # Grad Clipping
    grad_clip_norm: Optional[float] = hp.optional(
        default=None,
        doc='The norm to clip gradient magnitudes to. Default: None',
    )

    def initialize_object(self, model: ComposerModel, dataloader_hparams: DataLoaderHparams) -> FitKwargs:
        """Construct a kwargs dictionary that can be unpacked and passed into :meth:`.Trainer.fit`.

        Args:
            model (ComposerModel): The model.
            dataloader_hparams (DataLoaderHparams): The dataloader hyperparameters.

        Returns:
            FitKwargs: A kwargs dictionary that can be unpacked and passed into :meth:`.Trainer.fit`.
        """
        # Train DataLoader
        train_dataloader = _initialize_dataloader(
            dataset_hparams=self.train_dataset,
            dataloader_label=self.train_dataloader_label,
            batch_size=self.train_batch_size,
            subset_num_batches=self.train_subset_num_batches,
            dataloader_hparams=dataloader_hparams,
        )

        # Eval dataloader
        eval_dataloader = _initialize_eval_dataloader(
            model=model,
            eval_dataset_hparams=self.eval_dataset,
            evaluators=self.evaluators,
            eval_batch_size=self.eval_batch_size,
            eval_subset_num_batches=self.eval_subset_num_batches,
            dataloader_hparams=dataloader_hparams,
        )
        # Schedulers
        schedulers = None
        if self.schedulers is not None:
            schedulers = [scheduler.initialize_object() for scheduler in self.schedulers]

        return {
            "train_dataloader": train_dataloader,
            "train_dataloader_label": self.train_dataloader_label,
            "train_subset_num_batches": self.train_subset_num_batches,
            "compute_training_metrics": self.compute_training_metrics,
            "reset_time": self.reset_time,
            "duration": self.duration,
            "schedulers": schedulers,
            "scale_schedule_ratio": self.scale_schedule_ratio,
            "step_schedulers_every_batch": self.step_schedulers_every_batch,
            "eval_dataloader": eval_dataloader,
            "eval_subset_num_batches": self.eval_subset_num_batches,
            "eval_interval": self.eval_interval,
            "grad_accum": self.grad_accum,
            "precision": self.precision,
            "grad_clip_norm": self.grad_clip_norm,
        }


class EvalKwargs(TypedDict):
    """Typing annotation for kwargs that can be passed into :meth:`.Trainer.eval`.

    :meta private:
    """
    dataloader: Union[Iterable, DataSpec, dict]
    dataloader_label: str
    metrics: Union[Metric, MetricCollection]
    subset_num_batches: int
    log_level: Union[str, LogLevel]


@dataclasses.dataclass
class EvalHparams(hp.Hparams):

    hparams_registry = {
        "dataset": dataset_registry,
    }
    dataset: DatasetHparams = hp.required(doc="Validation dataset hparams")
    batch_size: int = hp.required(doc="batch size to use for each evaluation step")
    dataloader_label: str = hp.optional(doc="Dataloader label", default="eval")
    subset_num_batches: int = hp.optional(
        doc="If specified, stop each evaluation after this many batches.",
        default=-1,
    )
    log_level: LogLevel = hp.optional("The log level.", default=LogLevel.FIT)
    metric_names: Optional[List[str]] = hp.optional(
        doc=(
            "Name of the metrics for the evaluator. Can be a torchmetrics metric name or the "
            "class name of a metric returned by model.metrics(). If None (the default), uses all metrics in the model"),
        default=None,
    )

    def initialize_object(self, model: ComposerModel, dataloader_hparams: DataLoaderHparams) -> EvalKwargs:
        # Dataloader
        dataloader = _initialize_dataloader(
            dataset_hparams=self.dataset,
            dataloader_label=self.dataloader_label,
            batch_size=self.batch_size,
            subset_num_batches=self.subset_num_batches,
            dataloader_hparams=dataloader_hparams,
        )
        assert dataloader is not None, "The dataloader is a required argument"

        # Get and copy all the model's associated evaluation metrics
        model_metrics = model.metrics(train=False)
        if isinstance(model_metrics, Metric):
            # Forcing metrics to be a MetricCollection simplifies logging results
            model_metrics = MetricCollection([model_metrics])

        # Use all the metrics from the model if no metric_names are specified
        if self.metric_names is None:
            metrics = copy.deepcopy(model_metrics)
        else:
            metrics = MetricCollection([])
            for metric_name in self.metric_names:
                try:
                    metric = model_metrics[metric_name]
                except KeyError as e:
                    raise RuntimeError((f"No metric found with the name {metric_name}. Check if this "
                                        "metric is compatible/listed in your model metrics. ")) from e
                assert isinstance(metric, Metric), "all values of a MetricCollection.__getitem__ should be a metric"
                metrics.add_metrics(copy.deepcopy(metric))
            if len(metrics) == 0:
                raise RuntimeError(("No metrics compatible with your model were added to this evaluator. "
                                    "Check that the metrics you specified are compatible/listed in your model."))

        return {
            "dataloader": dataloader,
            "dataloader_label": self.dataloader_label,
            "metrics": metrics,
            "subset_num_batches": self.subset_num_batches,
            "log_level": self.log_level,
        }


@dataclasses.dataclass
class ExperimentHparams(hp.Hparams):
    trainer: TrainerHparams = hp.required("Trainer hparams")
    fits: List[FitHparams] = hp.optional("Fit hparams", default_factory=list)
    evals: List[EvalHparams] = hp.optional("Eval hparams", default_factory=list)

    def initialize_object(self) -> Tuple[Trainer, List[FitKwargs], List[EvalKwargs]]:
        """Construct the :class:`.Trainer`, :meth:`~Trainer.fit` kwargs, and
        :meth:`~Trainer.eval` kwargs.

        Returns:
            Tuple[Trainer, List[FitKwargs], List[EvalKwargs]]: A tuple of the
                (trainer, list of :meth:`~.Trainer.fit` kwargs, list of :meth:`~.Trainer.eval` kwargs).
        """
        trainer = self.trainer.initialize_object()
        # TODO(ravi): With MetricsModule, `fit_hparams` and `eval_hparams` will
        # no longer need the original model
        fit_kwargs = [
            fit_hparams.initialize_object(trainer._original_model, self.trainer.dataloader) for fit_hparams in self.fits
        ]
        eval_kwargs = [
            eval_hparams.initialize_object(trainer._original_model, self.trainer.dataloader)
            for eval_hparams in self.evals
        ]

        return trainer, fit_kwargs, eval_kwargs
