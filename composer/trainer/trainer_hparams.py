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
from composer.trainer.devices import Device, DeviceCPU, DeviceGPU
from composer.trainer.devices.device_hparams_registry import device_registry
from composer.trainer.trainer import Trainer
from composer.utils import dist, reproducibility
from composer.utils.object_store.object_store_hparams import ObjectStoreHparams, object_store_registry

if TYPE_CHECKING:
    from typing import TypedDict
else:
    TypedDict = object  # TypedDict is not available on python 3.7

# Specifically excluding `FitKwargs` and `EvalKwargs` from `__all__` and documentation
# They exist purely for pyright and should never need
__all__ = ['TrainerHparams', 'load', 'FitHparams', 'EvalHparams', 'ExperimentHparams']

Scheduler = Union[ComposerScheduler, PyTorchScheduler]

model_registry = {
    'unet': UnetHparams,
    'ssd': SSDHparams,
    'deeplabv3': DeepLabV3Hparams,
    'efficientnetb0': EfficientNetB0Hparams,
    'resnet_cifar': ResNetCIFARHparams,
    'resnet': ResNetHparams,
    'mnist_classifier': MnistClassifierHparams,
    'gpt2': GPT2Hparams,
    'bert': BERTHparams,
    'bert_classification': BERTForClassificationHparams,
    'timm': TimmHparams,
    'vit_small_patch16': ViTSmallPatch16Hparams
}


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
                f'The batch size for {dataloader_label} must be specified if the {dataloader_label} dataset is specified'
            )

        train_device_batch_size = batch_size // dist.get_world_size()
        if dataset_hparams.shuffle and subset_num_batches is not None:
            warnings.warn(
                (f'SubsetNumBatchesWarning: When specifying `subset_num_batches` for the {dataloader_label} dataset, '
                 f'dataset_hparams.shuffle should be set to False. '
                 'Otherwise, each epoch may load a different subset of samples.'))
        dataloader = dataset_hparams.initialize_object(train_device_batch_size, dataloader_hparams)
    return dataloader


def _parse_grad_accum(grad_accum: Union[int, str]) -> Union[int, str]:
    if grad_accum == 'auto':
        return grad_accum

    try:
        return int(grad_accum)
    except (ValueError, TypeError):
        pass

    raise ValueError('grad_accum should be "auto" or an integer.')


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
            'Either `eval_dataset` or `evaluators` should be specified. It is not permitted to specify both.')
    if eval_dataset_hparams is not None:
        eval_dataloader = _initialize_dataloader(
            eval_dataset_hparams,
            'eval',
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
                warnings.warn(('SubsetNumBatchesWarning: When specifying eval_subset_num_batches, '
                               f'(set to {eval_subset_num_batches}), evaluator.dataloader.shuffle '
                               f"(for Evaluator: '{evaluator.label}') should be set to False. Otherwise, "
                               'each evaluation epoch may load a different subset of samples.'))
    return eval_dataloader


@dataclasses.dataclass
class TrainerHparams(hp.Hparams):
    """Params for instantiating the :class:`.Trainer`.

    .. seealso:: The documentation for the :class:`.Trainer`.

    Args:
        model (ModelHparams): Hparams for constructing the model to train.

            .. seealso:: :mod:`composer.models` for models built into Composer.

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
        load_object_store (ObjectStoreHparams, optional): See :class:`.Trainer`. Both ``load_logger_destination`` and
            ``load_object_store`` should not be provided since there can only be one location to load from.
        load_logger_destination (LoggerDestination, optional): Used to specify a ``LoggerDestination`` for
            ``load_object_store`` in :class:`.Trainer` as Hparams doesn't support a Union type for those objects. Both
            ``load_logger_destination`` and ``load_object_store`` should not be provided since there can only be one location
            to load from.
        load_weights_only (bool, optional): See :class:`.Trainer`.
        load_strict_model_weights (bool, optional): See :class:`.Trainer`.
        load_progress_bar (bool, optional): See :class:`.Trainer`.
        load_ignore_keys (List[str] | (Dict) -> None, optional): See :class:`.Trainer`.

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
        autoresume (bool, optional): See :class:`.Trainer`.

        deepspeed_config (Dict[str, JSON], optional): If set to a dict will be used for as the DeepSpeed
            config for training  (see :class:`.Trainer` for more details). If ``None`` (the default), DeepSpeed will not
            be used.

        device (Device, optional): Hparams for constructing the device used for training.
            (default: ``None``)
        precision (Precision, optional): See :class:`.Trainer`.
        grad_accum (int | str, optional): See :class:`.Trainer`.

        seed (int, optional): See :class:`.Trainer`.
        deterministic_mode (bool, optional): See :class:`.Trainer`.

        dist_timeout (float, optional): See :class:`.Trainer`.
        ddp_sync_strategy (DDPSyncStrategy, optional): See :class:`.Trainer`.

        grad_clip_norm (float, optional): See :class:`.Trainer`.

        profiler (ProfilerHparams, optional): Profiler hyperparameters.
    """

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
        'load_object_store': object_store_registry,
    }

    model: ModelHparams = hp.auto(Trainer, 'model')

    # Shared data
    dataloader: DataLoaderHparams = hp.optional(doc='dataloader hparams', default=DataLoaderHparams())

    # Train Data
    train_dataset: Optional[DatasetHparams] = hp.optional(doc='Training dataset hparams', default=None)
    train_dataloader_label: str = hp.auto(Trainer, 'train_dataloader_label')
    train_batch_size: Optional[int] = hp.optional(
        doc='batch size for each optimization step, across all devices and gradient accumulations.',
        default=None,
    )
    train_subset_num_batches: int = hp.auto(Trainer, 'train_subset_num_batches')
    compute_training_metrics: bool = hp.auto(Trainer, 'compute_training_metrics')

    # Stopping Conditions
    max_duration: Optional[Union[str, int]] = hp.auto(Trainer, 'max_duration')

    # Algorithms
    algorithms: Optional[List[Algorithm]] = hp.auto(Trainer, 'algorithms')

    # Optimizer and Scheduler
    optimizers: Optional[OptimizerHparams] = hp.auto(Trainer, 'optimizers')
    schedulers: Optional[List[ComposerScheduler]] = hp.auto(Trainer, 'schedulers')
    scale_schedule_ratio: float = hp.auto(Trainer, 'scale_schedule_ratio')
    step_schedulers_every_batch: Optional[bool] = hp.auto(Trainer, 'step_schedulers_every_batch')

    # Evaluation
    val_dataset: Optional[DatasetHparams] = hp.optional(doc='Validation dataset hparams', default=None)
    evaluators: Optional[List[EvaluatorHparams]] = hp.optional(doc='Evaluators', default=None)
    eval_batch_size: Optional[int] = hp.optional(doc='batch size to use for each evaluation step', default=None)
    eval_interval: Union[int, str] = hp.auto(Trainer, 'eval_interval')
    eval_subset_num_batches: int = hp.auto(Trainer, 'eval_subset_num_batches')

    # Callbacks
    callbacks: Optional[List[Callback]] = hp.auto(Trainer, 'callbacks')

    # Logging
    loggers: Optional[List[LoggerDestination]] = hp.auto(Trainer, 'loggers')
    run_name: Optional[str] = hp.auto(Trainer, 'run_name')
    progress_bar: bool = hp.auto(Trainer, 'progress_bar')
    log_to_console: Optional[bool] = hp.auto(Trainer, 'log_to_console')
    console_log_level: LogLevel = hp.auto(Trainer, 'console_log_level')
    console_stream: str = hp.auto(Trainer, 'console_stream')
    python_log_level: str = hp.optional(doc='Python loglevel to use composer', default='INFO')

    # Load Checkpoint
    load_path: Optional[str] = hp.auto(Trainer, 'load_path')
    load_object_store: Optional[ObjectStoreHparams] = hp.optional(
        doc=(('If the checkpoint is in an object store (i.e. AWS S3 or Google Cloud Storage), the parameters for '
              'connecting to the cloud provider object store. Otherwise, if the checkpoint is a local filepath, '
              'leave blank. This parameter has no effect if `load_path` is not specified.')),
        default=None)
    load_logger_destination: Optional[LoggerDestination] = hp.optional(
        ('Alternative argument to `load_object_store` to support loading from a logger destination. This parameter '
         'has no effect if `load_path` is not specified or `load_object_store` is specified, which will be '
         'used instead of this.'),
        default=None)
    load_weights_only: bool = hp.auto(Trainer, 'load_weights_only')
    load_strict_model_weights: bool = hp.auto(Trainer, 'load_strict_model_weights')
    load_ignore_keys: Optional[List[str]] = hp.auto(Trainer, 'load_ignore_keys')
    load_progress_bar: bool = hp.auto(Trainer, 'load_progress_bar')

    # Save Checkpoint
    save_folder: Optional[str] = hp.auto(Trainer, 'save_folder')
    save_filename: str = hp.auto(Trainer, 'save_filename')
    save_artifact_name: str = hp.auto(Trainer, 'save_artifact_name')
    save_latest_filename: str = hp.auto(Trainer, 'save_latest_filename')
    save_latest_artifact_name: str = hp.auto(Trainer, 'save_latest_artifact_name')
    save_overwrite: bool = hp.auto(Trainer, 'save_overwrite')
    save_weights_only: bool = hp.auto(Trainer, 'save_weights_only')
    save_interval: str = hp.auto(Trainer, 'save_interval')
    save_num_checkpoints_to_keep: int = hp.auto(Trainer, 'save_num_checkpoints_to_keep')

    # Graceful Resumption
    autoresume: bool = hp.auto(Trainer, 'autoresume')
    # DeepSpeed
    deepspeed_config: Optional[Dict[str, JSON]] = hp.auto(Trainer, 'deepspeed_config')

    # System/Numerics
    device: Optional[Device] = hp.auto(Trainer, 'device')
    precision: Optional[Precision] = hp.auto(Trainer, 'precision')
    grad_accum: Union[int, str] = hp.auto(Trainer, 'grad_accum')

    # Reproducibility
    seed: Optional[int] = hp.auto(Trainer, 'seed')
    deterministic_mode: bool = hp.auto(Trainer, 'deterministic_mode')

    # Distributed
    dist_timeout: float = hp.auto(Trainer, 'dist_timeout')
    ddp_sync_strategy: Optional[DDPSyncStrategy] = hp.auto(Trainer, 'ddp_sync_strategy')

    # Grad Clip Norm
    grad_clip_norm: float = hp.auto(Trainer, 'grad_clip_norm')

    # Profiling
    profiler: Optional[Profiler] = hp.auto(Trainer, 'profiler')

    def validate(self):
        super().validate()

        if self.deepspeed_config is not None:
            self.deepspeed_config['steps_per_print'] = cast(int, self.deepspeed_config.get('steps_per_print', 1e20))

            if 'zero_optimization' in self.deepspeed_config:
                zero_stage = cast(dict, self.deepspeed_config['zero_optimization']).get('stage', 0)
            else:
                zero_stage = 0

            if self.deterministic_mode and zero_stage > 0:
                raise ValueError('Deepspeed with zero stage > 0 is not compatible with deterministic mode')

        world_size = dist.get_world_size()

        if self.train_batch_size is not None and self.train_batch_size % world_size != 0:
            raise ValueError(
                f'Batch size ({self.train_batch_size}) not divisible by the total number of processes ({world_size}).')

        val_dataset_exists = self.val_dataset is not None
        evaluators_exist = self.evaluators is not None and len(self.evaluators) > 0
        if val_dataset_exists and evaluators_exist:
            raise ValueError('Either val_dataset or evaluators should be set, but not both.')

        if (val_dataset_exists or evaluators_exist) and self.eval_batch_size is None:
            raise ValueError('eval_batch_size must be specified if val_dataset or evaluators are specified.')

        if self.eval_batch_size is not None and self.eval_batch_size % world_size != 0:
            raise ValueError(
                f'Eval batch size ({self.eval_batch_size}) not divisible by the total number of processes ({world_size}).'
            )

        if self.scale_schedule_ratio <= 0:
            raise ValueError('scale_schedule_ratio must be a positive value.')

        grad_accum = _parse_grad_accum(self.grad_accum)
        if (isinstance(grad_accum, str) and grad_accum != 'auto') or (isinstance(grad_accum, int) and grad_accum < 1):
            raise ValueError('grad_accum must be "auto" or an int greater than or equal to 1.')

    def initialize_object(self) -> Trainer:
        self.validate()

        # Set the Python LogLevel for Composer
        import composer
        logging.getLogger(composer.__name__).setLevel(self.python_log_level.upper())

        # ensure grad_accum is 'auto' or an integer
        grad_accum = _parse_grad_accum(self.grad_accum)

        # Device
        device = self.device
        if device is None:
            device = DeviceGPU() if torch.cuda.is_available() else DeviceCPU()

        # Distributed
        # Initialized here so it is available within dataloaders
        if dist.get_world_size() > 1:
            dist.initialize_dist(device, datetime.timedelta(seconds=self.dist_timeout))

        # Reproducibility
        seed = self.seed if self.seed else reproducibility.get_random_seed()
        # need to set seed before model initialization for determinism
        # don't need to set different seeds per process since only the rank 0 initialization is used
        # Algorithms should not use the `seed` on `__init__` but rather on `Event.INIT`, which occurs
        # after the seed was properly distributed across ranks to ensure checkpoint compatibility
        reproducibility.seed_all(seed)

        # The model
        model = self.model.initialize_object()

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

        # Optimizers
        optimizers = self.optimizers.initialize_object(model.parameters()) if self.optimizers is not None else None

        load_object_store = None
        if self.load_object_store is not None and self.load_logger_destination is not None:
            raise ValueError(
                'load_object_store and load_logger_destination cannot both be non-None. Please provide only one location to load from.'
            )
        elif self.load_object_store is not None:
            load_object_store = self.load_object_store.initialize_object()
        elif self.load_logger_destination is not None:
            load_object_store = self.load_logger_destination

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
            algorithms=self.algorithms,

            # Optimizers and Schedulers
            optimizers=optimizers,
            schedulers=self.schedulers,
            scale_schedule_ratio=self.scale_schedule_ratio,
            step_schedulers_every_batch=self.step_schedulers_every_batch,

            # Evaluation
            eval_dataloader=eval_dataloader,
            eval_interval=self.eval_interval,
            eval_subset_num_batches=self.eval_subset_num_batches,

            # Callbacks
            callbacks=self.callbacks,

            # Logging
            loggers=self.loggers,
            run_name=self.run_name,
            progress_bar=self.progress_bar,
            log_to_console=self.log_to_console,
            console_log_level=self.console_log_level,
            console_stream=self.console_stream,

            # Checkpoint Loading
            load_path=self.load_path,
            load_object_store=load_object_store,
            load_weights_only=self.load_weights_only,
            load_strict_model_weights=self.load_strict_model_weights,
            load_progress_bar=self.load_progress_bar,
            load_ignore_keys=self.load_ignore_keys,

            # Checkpoint Saving
            save_folder=self.save_folder,
            save_overwrite=self.save_overwrite,
            save_filename=self.save_filename,
            save_latest_filename=self.save_latest_filename,
            save_artifact_name=self.save_artifact_name,
            save_latest_artifact_name=self.save_latest_artifact_name,
            save_interval=self.save_interval,
            save_weights_only=self.save_weights_only,
            save_num_checkpoints_to_keep=self.save_num_checkpoints_to_keep,

            # Graceful Resumption
            autoresume=self.autoresume,

            # DeepSpeed
            deepspeed_config=self.deepspeed_config,

            # System/Numerics
            device=device,
            precision=self.precision,
            grad_accum=grad_accum,

            # Reproducibility
            seed=seed,
            deterministic_mode=self.deterministic_mode,

            # Distributed
            dist_timeout=self.dist_timeout,
            ddp_sync_strategy=self.ddp_sync_strategy,

            # Grad Clip Norm
            grad_clip_norm=self.grad_clip_norm,

            # Profiler
            profiler=self.profiler,
        )

        return trainer

    @classmethod
    def load(cls, model: str) -> TrainerHparams:
        model_hparams_file = os.path.join(
            os.path.dirname(composer.__file__),
            'yamls',
            'models',
            f'{model}.yaml',
        )
        trainer_hparams = TrainerHparams.create(model_hparams_file, cli_args=False)
        assert isinstance(trainer_hparams, TrainerHparams), 'trainer hparams should return an instance of self'
        return trainer_hparams


load = TrainerHparams.load


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
    """

    hparams_registry = {
        'train_dataset': dataset_registry,
        'schedulers': scheduler_registry,
        'eval_dataset': dataset_registry,
    }
    # Train dataloader
    train_dataset: Optional[DatasetHparams] = hp.required('Train dataset')
    train_batch_size: Optional[int] = hp.optional(
        doc='batch size for each optimization step, across all devices and gradient accumulations.',
        default=None,
    )
    train_dataloader_label: str = hp.auto(Trainer.fit, 'train_dataloader_label')
    train_subset_num_batches: Optional[int] = hp.auto(Trainer.fit, 'train_subset_num_batches')
    compute_training_metrics: Optional[bool] = hp.auto(Trainer.fit, 'compute_training_metrics')

    # Timing
    reset_time: bool = hp.auto(Trainer.fit, 'reset_time')
    duration: Optional[Union[int, str]] = hp.auto(Trainer.fit, 'duration')

    # Schedulers
    schedulers: Optional[List[ComposerScheduler]] = hp.auto(Trainer.fit, 'schedulers')
    scale_schedule_ratio: float = hp.optional(
        doc='Ratio by which to scale the training duration and learning rate schedules.',
        default=1.0,
    )
    step_schedulers_every_batch: Optional[bool] = hp.auto(Trainer.fit, 'step_schedulers_every_batch')

    # Evaluation
    eval_dataset: Optional[DatasetHparams] = hp.optional(doc='Validation dataset hparams', default=None)
    evaluators: Optional[List[EvaluatorHparams]] = hp.optional(doc='Evaluators', default=None)
    eval_batch_size: Optional[int] = hp.optional(doc='batch size to use for each evaluation step', default=None)
    eval_interval: Union[int, str] = hp.auto(Trainer.fit, 'eval_interval')
    eval_subset_num_batches: int = hp.auto(Trainer.fit, 'eval_subset_num_batches')

    # Numerics
    precision: Optional[Precision] = hp.auto(Trainer.fit, 'precision')
    grad_accum: Optional[Union[int, str]] = hp.auto(Trainer.fit, 'grad_accum')

    def initialize_object(self, model: ComposerModel, dataloader_hparams: DataLoaderHparams) -> FitKwargs:
        """Construct a kwargs dictionary that can be unpacked and passed into :meth:`.Trainer.fit`.

        Args:
            model (ComposerModel): The model.
            dataloader_hparams (DataLoaderHparams): The dataloader hyperparameters.

        Returns:
            FitKwargs: A kwargs dictionary that can be unpacked and passed into :meth:`.Trainer.fit`.
        """
        grad_accum = _parse_grad_accum(self.grad_accum) if self.grad_accum else self.grad_accum

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

        return {
            'train_dataloader': train_dataloader,
            'train_dataloader_label': self.train_dataloader_label,
            'train_subset_num_batches': self.train_subset_num_batches,
            'compute_training_metrics': self.compute_training_metrics,
            'reset_time': self.reset_time,
            'duration': self.duration,
            'schedulers': self.schedulers,
            'scale_schedule_ratio': self.scale_schedule_ratio,
            'step_schedulers_every_batch': self.step_schedulers_every_batch,
            'eval_dataloader': eval_dataloader,
            'eval_subset_num_batches': self.eval_subset_num_batches,
            'eval_interval': self.eval_interval,
            'grad_accum': grad_accum,
            'precision': self.precision,
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
    """Hyperparameters to describe a call to :meth:`.Trainer.eval`.

    Args:
        dataset (DatasetHparams): Dataset hyperparameters.
        batch_size (int): The evaluation batch size across all workers.
        dataloader_label (str, optional): See :meth:`.Trainer.eval`.
        subset_num_batches (int, optional): See :meth:`.Trainer.eval`.
        log_level (LogLevel, optional): See :meth:`.Trainer.eval`.
        metric_names (List[str], optional): Name of the metrics for the evaluator. (default: ``None``)

            Can be a :mod:`torchmetrics` metric name or the class name of a metric returned by
            :meth:`.ComposerModel.metrics`.
            If None (the default), uses all metrics in the model.
    """

    hparams_registry = {
        'dataset': dataset_registry,
    }
    dataset: DatasetHparams = hp.required(doc='Validation dataset hparams')
    batch_size: int = hp.required(doc='batch size to use for each evaluation step')
    dataloader_label: str = hp.auto(Trainer.eval, 'dataloader_label')
    subset_num_batches: int = hp.auto(Trainer.eval, 'subset_num_batches')
    log_level: LogLevel = hp.auto(Trainer.eval, 'log_level')
    metric_names: Optional[List[str]] = hp.optional(
        doc=(
            'Name of the metrics for the evaluator. Can be a torchmetrics metric name or the '
            'class name of a metric returned by model.metrics(). If None (the default), uses all metrics in the model'),
        default=None,
    )

    def initialize_object(self, model: ComposerModel, dataloader_hparams: DataLoaderHparams) -> EvalKwargs:
        """Construct a kwargs dictionary that can be unpacked and passed into :meth:`.Trainer.eval`.

        Args:
            model (ComposerModel): The model.
            dataloader_hparams (DataLoaderHparams): The dataloader hyperparameters.

        Returns:
            EvalKwargs: A kwargs dictionary that can be unpacked and passed into :meth:`.Trainer.eval`.
        """
        # Dataloader
        dataloader = _initialize_dataloader(
            dataset_hparams=self.dataset,
            dataloader_label=self.dataloader_label,
            batch_size=self.batch_size,
            subset_num_batches=self.subset_num_batches,
            dataloader_hparams=dataloader_hparams,
        )
        assert dataloader is not None, 'The dataloader is a required argument'

        # Metrics

        # TODO(Ravi): Cleanup this code as part of the MetricsModule. This code was copied
        # from composer/datasets/evaluator.py, but will likely be removed when the MetricsModule
        # is implemented, as the trainer will not be responsible for constructing metrics.

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
                    raise RuntimeError((f'No metric found with the name {metric_name}. Check if this '
                                        'metric is compatible/listed in your model metrics. ')) from e
                assert isinstance(metric, Metric), 'all values of a MetricCollection.__getitem__ should be a metric'
                metrics.add_metrics(copy.deepcopy(metric))
            if len(metrics) == 0:
                raise RuntimeError(('No metrics compatible with your model were added to this evaluator. '
                                    'Check that the metrics you specified are compatible/listed in your model.'))

        return {
            'dataloader': dataloader,
            'dataloader_label': self.dataloader_label,
            'metrics': metrics,
            'subset_num_batches': self.subset_num_batches,
            'log_level': self.log_level,
        }


@dataclasses.dataclass
class ExperimentHparams(hp.Hparams):
    """Hyperparameters to describe an experiment.

    Unlike the :class:`.TrainerHparams`, this class allows for multiple configurations
    to :meth:`.Trainer.fit` and :meth:`.Trainer.eval` to also be specified.

    Example usage:

    .. testsetup::

        from tests.common import SimpleModelHparams, RandomClassificationDatasetHparams

        from composer.datasets.dataset_hparams import DataLoaderHparams
        from composer.trainer.trainer_hparams import ExperimentHparams, FitHparams, EvalHparams, TrainerHparams

        trainer_hparams = TrainerHparams(
            model=SimpleModelHparams(),
            dataloader=DataLoaderHparams(
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
            ),
            max_duration=1,
        )
        fit_1_hparams = FitHparams(
            train_dataset=RandomClassificationDatasetHparams(),
            train_batch_size=1,
            train_subset_num_batches=1,
            reset_time=True,
        )
        fit_2_hparams = fit_1_hparams
        eval_1_hparams = EvalHparams(
            dataset=RandomClassificationDatasetHparams(),
            batch_size=1,
            subset_num_batches=1,
        )
        eval_2_hparams = eval_1_hparams

    .. testcode::

        experiment_hparams = ExperimentHparams(
            trainer=trainer_hparams,
            fits=[fit_1_hparams, fit_2_hparams],
            evals=[eval_1_hparams, eval_2_hparams],
        )

        trainer, fits, evals = experiment_hparams.initialize_object()

        # The caller can invoke `trainer.fit(...)` and `trainer.eval(...)` in whatever
        # order or combination makes sense for the experiment.
        # In this example, all of the evaluations are run for each call to `trainer.fit(...)`.

        for fit_kwargs in fits:
            trainer.fit(**fit_kwargs)
            for eval_kwargs in evals:
                trainer.eval(**eval_kwargs)

    Args:
        trainer (TrainerHparams): The trainer hparams.
        fits (List[FitHparams]): The hparams for calls to :meth:`.Trainer.fit`.
        evals (List[EvalHparams]): The hparams for calls to :meth:`.Trainer.eval`.
    """
    trainer: TrainerHparams = hp.required('Trainer hparams')
    fits: List[FitHparams] = hp.optional('Fit hparams', default_factory=list)
    evals: List[EvalHparams] = hp.optional('Eval hparams', default_factory=list)

    def initialize_object(self) -> Tuple[Trainer, List[FitKwargs], List[EvalKwargs]]:
        """Construct the :class:`.Trainer`, :meth:`~Trainer.fit` kwargs, and :meth:`~Trainer.eval` kwargs.

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
