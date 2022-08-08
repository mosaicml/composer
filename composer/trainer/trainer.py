# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Train models."""

from __future__ import annotations

import contextlib
import datetime
import itertools
import logging
import os
import pathlib
import time
import warnings
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union, cast

import coolname
import torch
import torch.distributed
import torch.nn as nn
import torch.utils.data
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Metric, MetricCollection

from composer.algorithms import GradientClipping
from composer.callbacks import CheckpointSaver
from composer.core import (Algorithm, Callback, DataSpec, Engine, Evaluator, Event, Precision, State, Time, Timestamp,
                           ensure_data_spec, ensure_evaluator, ensure_time)
from composer.core.precision import get_precision_context
from composer.core.time import TimeUnit
from composer.core.types import Batch, PyTorchScheduler
from composer.loggers import Logger, LoggerDestination, LogLevel, ProgressBarLogger
from composer.models.base import ComposerModel
from composer.optim.decoupled_weight_decay import DecoupledSGDW
from composer.optim.scheduler import ComposerScheduler, compile_composer_scheduler
from composer.profiler import Profiler
from composer.trainer._deepspeed import _fix_batch_precision_for_deepspeed, _parse_deepspeed_config
from composer.trainer._scale_schedule import scale_pytorch_scheduler
from composer.trainer._scaler import ClosureGradScaler
from composer.trainer.ddp import DDPSyncStrategy, ddp_sync_context, prepare_ddp_module
from composer.trainer.devices import Device, DeviceCPU, DeviceGPU
from composer.utils import (ObjectStore, dist, ensure_tuple, format_name_with_dist, is_model_deepspeed, map_collection,
                            reproducibility)
from composer.utils.checkpoint import load_checkpoint, save_checkpoint
from composer.utils.file_helpers import get_file
from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.inference import ExportFormat, Transform, export_with_logger

log = logging.getLogger(__name__)

__all__ = ['Trainer']

# syntax to shorten the Scheduler type annoations
Scheduler = Union[ComposerScheduler, PyTorchScheduler]


def _raise_missing_argument_exception(arg_name: str):
    raise ValueError((f'{arg_name} is a required argument and must be specified when constructing the '
                      f'{Trainer.__name__} or when calling {Trainer.__name__}.{Trainer.fit.__name__}(). '
                      f'To fix, please specify `{arg_name}` via {Trainer.__name__}({arg_name}=...) or '
                      f'{Trainer.__name__}.{Trainer.fit.__name__}({arg_name}=...).'))


def _scale_max_duration_by_ssr(
    scale_schedule_ratio: float,
    orig_max_duration: Optional[Time[int]],
) -> Optional[Time[int]]:
    if orig_max_duration is None:
        return None
    max_duration = cast(Time[int], orig_max_duration * scale_schedule_ratio)
    log.info(f'max_duration changed from {orig_max_duration} to {max_duration}')
    if max_duration.value == 0:
        raise ValueError('Scale schedule has reduced the max_duration to 0. Set a higher ratio or use more epochs.')
    return max_duration


def _get_default_scheduler_frequency(schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]]):
    has_pytorch_scheduler = any(isinstance(scheduler, PyTorchScheduler) for scheduler in ensure_tuple(schedulers))
    if has_pytorch_scheduler:
        log.info(('Stepping schedulers every epoch, as a PyTorch scheduler was provided. '
                  'The trainer cannot automatically convert the parameters (e.g. step_size, T_max) of the '
                  'PyTorch scheduler to be in terms of batches. If the PyTorch scheduler should be stepped '
                  'every batch, set `step_schedulers_every_batch=True`.'))
        return TimeUnit.EPOCH
    else:
        log.info(('Stepping schedulers every batch. '
                  'To step schedulers every epoch, set `step_schedulers_every_batch=False`.'))
        return TimeUnit.BATCH


def _get_training_metrics(model: ComposerModel):
    warnings.warn(('Computing model evaluation metrics during training doubles the number of forward passes '
                   'and may lead to a throughput degradation.'))
    train_metrics = model.metrics(train=True)
    if isinstance(train_metrics, Metric):
        # Forcing metrics to be a MetricCollection simplifies logging results
        train_metrics = MetricCollection([train_metrics])

    return train_metrics


def _validate_precision(precision: Precision, device: Device, deepspeed_enabled: bool):
    if isinstance(device, DeviceCPU) and precision != Precision.FP32:
        raise ValueError(f'{precision} is not supproted for CPU training.')
    if not deepspeed_enabled and precision == Precision.FP16:
        raise ValueError('FP16 precision is only supported when training with DeepSpeed.')


def _compile_schedulers(
    schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]],
    state: State,
    scale_schedule_ratio: float,
) -> List[PyTorchScheduler]:
    compiled_schedulers = []
    for scheduler in ensure_tuple(schedulers):
        if isinstance(scheduler, PyTorchScheduler):
            scale_pytorch_scheduler(scheduler, scale_schedule_ratio)
            compiled_schedulers.append(scheduler)
        else:  # it's a composer scheduler
            compiled_schedulers.append(compile_composer_scheduler(scheduler, state, scale_schedule_ratio))

    return compiled_schedulers


def _set_evaluator_interval_and_subset_num_batches(
    evaluators: Sequence[Evaluator],
    eval_interval: Union[int, str, Time, Callable[[State, Event], bool]],
    subset_num_batches: int,
):
    # convert eval_dataloader to `List[Evaluator]`
    for evaluator in evaluators:
        if evaluator.subset_num_batches is None:
            evaluator.subset_num_batches = subset_num_batches
        if evaluator.eval_interval is None:
            evaluator.eval_interval = eval_interval


def _is_adaptive_grad_accum(grad_accum: Union[int, str], device: Device):
    if grad_accum == 'auto':
        warnings.warn(("Setting `grad_accum='auto'` is an experimental feature which may cause "
                       'uncaught Cuda Out of Memory errors. In this case, please manually '
                       'set grad_accum explicitly to an integer instead. '))
        if isinstance(device, DeviceCPU):
            raise ValueError('Cannot use adaptive grad_accum on CPU. Please set grad_accum >= 1')
        return True
    else:
        return False


def _get_initial_grad_accum(grad_accum: Union[int, str]):
    if grad_accum == 'auto':
        return 1
    elif isinstance(grad_accum, int):
        return grad_accum
    else:
        raise ValueError("grad_accum must be an int or ``'auto'``")


def _is_cuda_oom(e: RuntimeError):
    """Determines if error is CUDA Out of Memory and if adaptive_grad_accum is enabled."""
    return 'CUDA out of memory' in str(e)


def _get_device(device: Optional[Union[str, Device]]):
    if not device:
        device = DeviceGPU() if torch.cuda.is_available() else DeviceCPU()
    elif isinstance(device, str):
        if device.lower() == 'cpu':
            device = DeviceCPU()
        elif device.lower() == 'gpu':
            device = DeviceGPU()
        else:
            raise ValueError(f'device ({device}) must be one of (cpu, gpu).')
    return device


def _distribute_and_get_random_seed(seed: Optional[int], device: Device):
    if not seed:
        seed = reproducibility.get_random_seed()

    # Ensure that each process has a seed = rank_zero_seed + global_rank
    # This "deterministically different" seed behavior is required to be able
    # to restore seeds when resuming form checkpoints, since only the
    # `rank_zero_seed` is stored on state.
    if seed < 0 or seed > reproducibility.MAX_SEED:
        raise ValueError(f'Invalid seed: {seed}. It must be on [0; 2**32 - 1)')

    # using int64 to prevent overflow
    rank_zero_seed = device.tensor_to_device(torch.tensor([seed], dtype=torch.int64))
    dist.broadcast(rank_zero_seed, src=0)
    rank_zero_seed = rank_zero_seed.item()
    assert isinstance(rank_zero_seed, int)
    seed = rank_zero_seed + dist.get_global_rank()
    return rank_zero_seed, seed


def _get_ddp_sync_strategy(ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]], find_unused_parameters: bool):
    if ddp_sync_strategy is None:
        if find_unused_parameters:
            ddp_sync_strategy = DDPSyncStrategy.MULTI_AUTO_SYNC
        else:
            ddp_sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC
    else:
        ddp_sync_strategy = DDPSyncStrategy(ddp_sync_strategy)
    return ddp_sync_strategy


def _generate_run_name() -> str:
    # prefixing with the time so experiments sorted alphabetically will have the latest experiment last
    generated_run_name = str(int(time.time())) + '-' + coolname.generate_slug(2)
    run_name_list = [generated_run_name]
    # ensure all ranks have the same experiment name
    dist.broadcast_object_list(run_name_list)
    generated_run_name = run_name_list[0]
    return generated_run_name


class Trainer:
    """Train models with Composer algorithms.

    The trainer supports models with :class:`~composer.models.base.ComposerModel` instances.
    The :class:`.Trainer` is highly customizable and can support a wide variety of workloads.
    See the :doc:`training guide</trainer/using_the_trainer>` for more information.

    Example
    --------

    Train a model and save a checkpoint:

    .. testcode::

        import os
        from composer import Trainer

        ### Create a trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration="1ep",
            eval_dataloader=eval_dataloader,
            optimizers=optimizer,
            schedulers=scheduler,
            device="cpu",
            eval_interval="1ep",
            save_folder="checkpoints",
            save_filename="ep{epoch}.pt",
            save_interval="1ep",
            save_overwrite=True,
        )

        # Fit and run evaluation for 1 epoch.
        # Save a checkpoint after 1 epoch as specified during trainer creation.
        trainer.fit()

    Load the checkpoint and resume training:

    .. testcode::

        # Get the saved checkpoint filepath
        checkpoint_path = trainer.saved_checkpoints.pop()[0]

        # Create a new trainer with the `load_path` argument set to the checkpoint path.
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration="2ep",
            eval_dataloader=eval_dataloader,
            optimizers=optimizer,
            schedulers=scheduler,
            device="cpu",
            eval_interval="1ep",
            load_path=checkpoint_path,
        )

        # Continue training and running evaluation where the previous trainer left off
        # until the new max_duration is reached.
        # In this case it will be one additional epoch to reach 2 epochs total.
        trainer.fit()


    Args:
        model (ComposerModel): The model to train. Can be user-defined or one of the models included
            with Composer.

            .. seealso:: :mod:`composer.models` for models built into Composer.
        train_dataloader (Iterable | DataSpec | dict, optional): The dataloader, :class:`.DataSpec`,
            or dict of :class:`.DataSpec` kwargs for the training data. In order to specify custom
            preprocessing steps on each data batch, specify a :class:`.DataSpec` instead of a dataloader.
            It is recommended that the dataloader, whether specified directly or as part of a :class:`.DataSpec`,
            should be a :class:`torch.utils.data.DataLoader`.

            .. note:: The ``train_dataloader`` should yield per-rank batches. Each per-rank batch
                will then be further divided based on the ``grad_accum`` parameter. For example, if the
                desired optimization batch size is ``2048`` and training is happening across 8 GPUs, then each
                ``train_dataloader`` should yield a batch of size ``2048 / 8 = 256``. If ``grad_accum = 2``,
                then the per-rank batch will be divided into microbatches of size ``256 / 2 = 128``.

            If ``train_dataloader`` is not specified when constructing the trainer, it must be specified when invoking
            :meth:`.Trainer.fit`.
        train_dataloader_label (str, optional): The label for the train dataloader. (default: ``'train'``)

            This label is used to index the training metrics (if ``compute_training_metrics`` is True) in
            :attr:`.State.current_metrics`.

            This parameter has no effect if ``train_dataloader`` or ``compute_training_metrics`` are not specified.
        train_subset_num_batches (int, optional): If specified, finish every epoch early after training
            on this many batches. This parameter has no effect if it is greater than ``len(train_dataloader)``.
            If ``-1``, then the entire dataloader will be iterated over. (default: ``-1``)

            When using the profiler, it can be helpful to set this parameter to the length of the profile schedule.
            This setting will end each epoch early to avoid additional training that will not be profiled.

            This parameter is ignored if ``train_dataloader`` is not specified.
        compute_training_metrics (bool, optional): Whether to compute training metrics. (default: ``False``)

            Training metrics will be indexed on :attr:`.State.current_metrics` under the ``train_dataloader_label``
            key (which defaults to ``'train'``).
        max_duration (Time | str | int, optional): The maximum duration to train. Can be an integer, which will be
            interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), or a :class:`.Time` object.

            If ``max_duration`` is not specified when constructing the trainer, ``duration`` must be specified when invoking
            :meth:`.Trainer.fit`.
        algorithms (Algorithm | Sequence[Algorithm], optional): The algorithms to use during training. If ``None``, then
            no algorithms will be used. (default: ``None``)

            .. seealso:: :mod:`composer.algorithms` for the different algorithms built into Composer.
        optimizers (torch.optim.Optimizer, optional): The optimizer.
            If ``None``, will be set to ``DecoupledSGDW(model.parameters(), lr=0.1)``. (default: ``None``)

            .. seealso:: :mod:`composer.optim` for the different optimizers built into Composer.
        schedulers (PyTorchScheduler | ComposerScheduler | Sequence[PyTorchScheduler | ComposerScheduler], optional):
            The learning rate schedulers. If ``[]`` or ``None``, the learning rate will be constant.
            (default: ``None``).

            .. seealso:: :mod:`composer.optim.scheduler` for the different schedulers built into Composer.
        scale_schedule_ratio (float, optional): Ratio by which to scale the training duration and learning rate
            schedules. (default: ``1.0``)

            E.g., ``0.5`` makes the schedule take half as many epochs and ``2.0`` makes it take twice as
            many epochs. ``1.0`` means no change.

            This parameter has no effect if ``schedulers`` is not specified.

            .. note ::

                Training for less time, while rescaling the learning rate schedule,
                is a strong baseline approach to speeding up training. E.g., training
                for half duration often yields minor accuracy degradation,
                provided that the learning rate schedule is also rescaled to take half as long.

                To see the difference, consider training for half as long using a cosine
                annealing learning rate schedule. If the schedule is not rescaled,
                training ends while the learning rate is still ~0.5 of the initial LR.
                If the schedule is rescaled with ``scale_schedule_ratio``, the LR schedule
                would finish the entire cosine curve, ending with a learning rate near zero.
        step_schedulers_every_batch (bool, optional): By default, native
            `PyTorch schedulers <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
            are updated every epoch, while :doc:`Composer Schedulers</trainer/schedulers>` are updated every step.
            Setting this to ``True`` will force schedulers to be stepped every batch,
            while ``False`` means schedulers stepped every epoch. ``None`` indicates the default behavior.
            (default: ``None``)
        eval_dataloader (DataLoader | DataSpec | Evaluator | Sequence[Evaluator], optional): The :class:`.DataLoader`,
            :class:`.DataSpec`, :class:`.Evaluator`, or sequence of evaluators for the evaluation data.

            To evaluate one or more specific metrics across one or more datasets, pass in an
            :class:`.Evaluator`. If a :class:`.DataSpec` or :class:`.DataLoader` is passed in, then all
            metrics returned by ``model.metrics()`` will be used during evaluation.
            ``None`` results in no evaluation. (default: ``None``)
        eval_interval (int | str | Time | (State, Event) -> bool, optional): Specifies how frequently to run evaluation.
            An integer, which will be interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), a :class:`.Time`
            object, or a callable.
            Defaults to ``1`` (evaluate every epoch).

            If an integer (in epochs), :class:`.Time` string, or :class:`.Time` instance, the evaluator will be run
            with this frequency. :class:`.Time` strings or :class:`.Time` instances must have units of
            :attr:`.TimeUnit.BATCH` or :attr:`.TimeUnit.EPOCH`.

            Set to ``0`` to disable evaluation.

            If a callable, it should take two arguments (:class:`.State`, :class:`.Event`) and return a bool
            representing whether the evaluator should be invoked. The event will be either :attr:`.Event.BATCH_END`
            or :attr:`.Event.EPOCH_END`.

            This ``eval_interval`` will apply to any :class:`.Evaluator` in ``eval_dataloader`` that does not specify
            an ``eval_interval`` or if a dataloader is passed in directly. This parameter has no effect if
            ``eval_dataloader`` is not specified.

            When specifying time string or integer for the ``eval_interval``, the evaluator(s) are also run at the ``Event.FIT_END`` if it doesn't
            evenly divide the training duration.

        eval_subset_num_batches (int, optional): If specified, evaluate on this many batches. Defaults to ``-1``,
            which means to iterate over the entire dataloader.

            This parameter has no effect if ``eval_dataloader`` is not specified, it is greater than
            ``len(eval_dataloader)``, or ``eval_dataloader`` is an :class:`.Evaluator` and ``subset_num_batches``
            was specified as part of the :class:`.Evaluator`.
        callbacks (Callback | Sequence[Callback], optional): The callbacks to run during training. If ``None``,
            then no callbacks will be run. (default: ``None``).

            .. seealso:: :mod:`composer.callbacks` for the different callbacks built into Composer.
        loggers (LoggerDestination | Sequence[LoggerDestination], optional): The destinations to log training information to.

            .. seealso:: :mod:`composer.loggers` for the different loggers built into Composer.
        run_name (str, optional): A name for this training run. If not specified, the timestamp will be combined with a
            :doc:`coolname <coolname:index>`, e.g. ``1654298855-electric-zebra``.
        progress_bar (bool, optional): Whether to show a progress bar. (default: ``True``)
        log_to_console (bool, optional): Whether to print logging statements to the console. (default: ``None``)

            The default behavior (when set to ``None``) only prints logging statements when ``progress_bar`` is ``False``.

        console_log_level (LogLevel | str | (State, LogLevel) -> bool, optional): The maximum log level which
            should be printed to the console. (default: :attr:`.LogLevel.EPOCH`)

            It can either be :class:`.LogLevel`, a string corresponding to a :class:`.LogLevel`, or a callable
            that takes the training :class:`.State` and the :class:`.LogLevel` and returns a boolean of whether this
            statement should be printed.

            This parameter has no effect if ``log_to_console`` is ``False``, or is unspecified and ``progres_bar`` is
            ``True``.

        console_stream (TextIO | str, optional): The stream to write to. If a string, it can either be
            ``'stdout'`` or ``'stderr'``. (default: :attr:`sys.stderr`)
        load_path (str, optional):  The path format string to an existing checkpoint file.

            It can be a path to a file on the local disk, a URL, or if ``load_object_store`` is set, the object name
            for a checkpoint in a cloud bucket.

            When using `Deepspeed ZeRO <https://www.deepspeed.ai/tutorials/zero/>`_, checkpoints are shareded by rank.
            Instead of hard-coding the rank in the ``path``, use the following format variables:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~.dist.get_global_rank`.                       |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~.dist.get_local_rank`.                        |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~.dist.get_node_rank`.                         |
            +------------------------+-------------------------------------------------------+

            For example, suppose that checkpoints are stored in the following structure:

            .. code-block::

                my_model/ep1-rank0.tar
                my_model/ep1-rank1.tar
                my_model/ep1-rank2.tar
                ...

            Then, ``load_path`` should be set to ``my_model/ep1-rank{rank}.tar``, and all ranks will load the
            correct state.

            If ``None`` then no checkpoint will be loaded. (default: ``None``)
        load_object_store (Union[ObjectStore, LoggerDestination], optional): If the ``load_path`` is in an
            object store (i.e. AWS S3 or Google Cloud Storage), an instance of :class:`.ObjectStore` or
            :class:`.LoggerDestination` which will be used to retreive the checkpoint. Otherwise, if the
            checkpoint is a local filepath, set to ``None``. Ignored if ``load_path`` is ``None``.
            (default: ``None``)

            Example:

            .. testsetup::

                import composer.trainer

                composer.trainer.trainer.load_checkpoint = lambda *args, **kwargs: None

            .. testcode::

                from composer import Trainer
                from composer.utils import LibcloudObjectStore

                # Create the object store provider with the specified credentials
                creds = {"key": "object_store_key",
                         "secret": "object_store_secret"}
                store = LibcloudObjectStore(provider="s3",
                                            container="my_container",
                                            provider_kwargs=creds)

                checkpoint_path = "./path_to_the_checkpoint_in_object_store"

                # Create a trainer which will load a checkpoint from the specified object store
                trainer = Trainer(
                    model=model,
                    train_dataloader=train_dataloader,
                    max_duration="10ep",
                    eval_dataloader=eval_dataloader,
                    optimizers=optimizer,
                    schedulers=scheduler,
                    device="cpu",
                    eval_interval="1ep",
                    load_path=checkpoint_path,
                    load_object_store=store,
                )
        load_weights_only (bool, optional): Whether or not to only restore the weights from the checkpoint without
            restoring the associated state. Ignored if ``load_path`` is ``None``. (default: ``False``)
        load_strict_model_weights (bool, optional): Ensure that the set of weights in the checkpoint and model must exactly match.
            Ignored if ``load_path`` is ``None``. (default: ``False``)
        load_progress_bar (bool, optional): Display the progress bar for downloading the checkpoint.
            Ignored if ``load_path`` is either ``None`` or a local file path. (default: ``True``)
        load_ignore_keys (List[str] | (Dict) -> None, optional): A list of paths for the ``state_dict`` of the checkpoint,
            which, when provided, will be ignored from the state_dict before a checkpoint is loaded. Each path is a list
            of strings specifying the keys to index into ``state_dict`` joined together with `/` as a seperator (as PyTorch
            uses `.` in parameter names). If a prefix is provided, all children are also ignored (see Example 2).
            See :mod:`composer.core.state` for the structure of state_dict.

            Example 1: ``load_ignore_keys = ["state/model/layer1.weights", "state/model/layer1.bias"]`` would ignore
            layer 1 weights and bias.

            Example 2: ``load_ignore_keys = ["state/model/*"]`` would ignore the entire model, which would have the same
            effect as the previous example if there was only 1 layer.

            Example 3: ``load_ignore_keys = ["state/model/layer*.weights"]`` would ignore all weights in the model.

            Example 4: ``load_ignore_keys = ["state/rank_zero_seed", "rng"]`` would reset all randomness when
            loading the checkpoint.

            If a callable, it should take one argument which is the state_dict. The callable is free to arbitrarily modify
            the state_dict before it is loaded.

            (default: ``None``)

        save_folder (str, optional): Format string for the folder where checkpoints are saved.
            If ``None``, checkpoints will not be saved. (default: ``None``)

            .. seealso:: :class:`~.CheckpointSaver`

            .. note::

                For fine-grained control on checkpoint saving (e.g. to save different types of checkpoints
                at different intervals), leave this parameter as ``None``, and instead pass
                instance(s) of :class:`~.CheckpointSaver` directly as ``callbacks``.
        save_filename (str, optional): A format string describing how to name checkpoints.
            This parameter has no effect if ``save_folder`` is ``None``.
            (default: ``"ep{epoch}-ba{batch}-rank{rank}.pt"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_artifact_name (str, optional): A format string describing how to name checkpoints in loggers.
            This parameter has no effect if ``save_folder`` is ``None``.
            (default: ``"{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_latest_filename (str, optional): A format string for the name of a symlink
            (relative to ``save_folder``) that points to the last saved checkpoint.
            This parameter has no effect if ``save_folder`` is ``None``.
            To disable symlinking, set this to ``None``. (default: ``"latest-rank{rank}.pt"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_latest_artifact_name (str, optional): A format string describing how to name symlinks in loggers.
            This parameter has no effect if ``save_folder``, ``save_latest_filename``, or ``save_artifact_name`` are ``None``.
            To disable symlinking in logger, set this or ``save_latest_filename`` to ``None``. (default: ``"{run_name}/checkpoints/latest-rank{rank}"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_overwrite (bool, optional): Whether existing checkpoints should be overridden.
            This parameter has no effect if ``save_folder`` is None. (default: ``False``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_interval (Time | str | int | (State, Event) -> bool): A :class:`Time`, time-string, integer (in epochs),
            or a function that takes (state, event) and returns a boolean whether a checkpoint should be saved.
            This parameter has no effect if ``save_folder`` is ``None``. (default: ``'1ep'``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_weights_only (bool, optional): Whether to save only the model weights instead of the entire training
            state. This parameter has no effect if ``save_folder`` is ``None``. (default: ``False``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_num_checkpoints_to_keep (int, optional): The number of checkpoints to keep locally. The oldest checkpoints
            are removed first. Set to ``-1`` to keep all checkpoints locally. (default: ``-1``)

            Checkpoints will be removed after they have been logged as a file artifact. For example, when this callback
            is used in conjunction with the :class:`~composer.loggers.object_store_logger.ObjectStoreLogger`, set this
            parameter to ``0`` to immediately delete checkpoints from the local disk after they have been uploaded to
            the object store.

            This parameter only controls how many checkpoints are kept locally; checkpoints are not deleted from
            artifact stores.
        autoresume (bool, optional): Whether or not to enable autoresume, which allows for stopping and resuming
        training. This allows use of spot instances, as the training run is now fault tolerant.  This parameter requires
            ``save_folder`` and ``run_name`` to be specified and ``save_overwrite`` to be ``False``. (default: ``False``)

            When enabled, the save_folder is checked for checkpoints of the format ``"{save_folder}/{save_latest_filename}"``,
            which are loaded to continue training. If no local checkpoints are found, each logger is checked for potential
            checkpoints named ``save_latest_artifact_name``. Finally, if no logged checkpoints are found, ``load_path`` is
            used to load a checkpoint if specified. This should only occur at the start of a run using autoresume.

            For example, to run a fine-tuning run on a spot instance, ``load_path`` would be set to the original weights and
            an object store logger would be added. In the original run, ``load_path`` would be used to get the starting
            checkpoint. For any future restarts, such as due to the spot instance being killed, the loggers would be queried for the latest checkpoint
            the object store logger would be downloaded and used to resume training.
        deepspeed_config (Dict[str, Any], optional): Configuration for DeepSpeed, formatted as a JSON
            according to `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_. (default: ``None``)

            To use DeepSpeed with default values, set to the empty dictionary ``{}``.
            To disable DeepSpeed (the default), set to ``None``.
        device (Device | str, optional): The device to use for training, which can be ``'cpu'`` or ``'gpu'``.
            (default: ``None``)

            The default behavior sets the device to ``'gpu'`` if CUDA is available; otherwise, it sets the device to
            ``'cpu'``.
        precision (Precision | str, optional): Numerical precision to use for training. One of ``fp32``, ``fp16``
            or ``amp`` (recommended). (default: ``Precision.FP32`` if training on CPU; ``Precision.AMP`` if training
            on GPU)

            .. note::
                ``fp16`` only works if ``deepspeed_config`` is also provided.
        grad_accum (Union[int, str], optional): The number of microbatches to split a per-device batch into. Gradients
            are summed over the microbatches per device. If set to ``auto``, dynamically increases grad_accum
            if microbatch is too large for GPU. (default: ``1``)

            .. note:: This is implemented by taking the batch yielded by the ``train_dataloader`` and splitting
                it into ``grad_accum`` sections. Each section is of size ``train_dataloader // grad_accum``.
                If the batch size of the dataloader is not divisible by ``grad_accum``,
                then the last section will be of size ``batch_size mod grad_accum``.
        seed (int, optional): The seed used in randomization. If ``None``, then a random seed
            will be created. (default: ``None``)

            .. note:: In order to get reproducible results, call the
                :func:`.seed_all` function at the start of your script with the seed
                passed to the trainer. This will ensure any initialization done before the trainer init
                (ex. model weight initialization) also uses the provided seed.

            .. seealso:: :mod:`composer.utils.reproducibility` for more details on reproducibility.
        deterministic_mode (bool, optional): Run the model deterministically. (default: ``False``)

            .. note:: This is an experimental feature. Performance degradations expected. Certain Torch modules may
                not have deterministic implementations, which will result in a crash.

            .. note:: In order to get reproducible results, call the
                :func:`.configure_deterministic_mode` function at the start of your script.
                This will ensure any initialization done before the trainer init also runs deterministically.

            .. seealso:: :mod:`composer.utils.reproducibility` for more details on reproducibility.
        dist_timeout (float, optional): Timeout, in seconds, for initializing the distributed process group.
            (default: ``15.0``)
        ddp_sync_strategy (str | DDPSyncStrategy, optional): The strategy to use for synchronizing gradients.
            Leave unset to let the trainer auto-configure this. See :class:`.DDPSyncStrategy`
            for more details.
        grad_clip_norm (float, optional): The norm to clip gradient magnitudes to. Set to ``-1`` for no gradient
            clipping. (default: ``-1``).

            .. deprecated:: 0.8
               Deprecated. Please use composer.algorithms.GradientClipping.
        profiler (Profiler, optional): The profiler, if profiling should be enabled. (default: ``None``)

            .. seealso::

                See the :doc:`Profiling Guide </trainer/performance_tutorials/profiling>` for
                additional information.

    Attributes:
        state (State): The :class:`.State` object used to store training state.
        evaluators (List[Evaluator]): The :class:`.Evaluator` objects to use for validation
            during training.
        logger (Logger): The :class:`.Logger` used for logging.
        engine (Engine): The :class:`.Engine` used for running callbacks and algorithms.
    """

    def __init__(
        self,
        *,
        # The Model
        model: ComposerModel,

        # Train Dataloader
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = 'train',
        train_subset_num_batches: int = -1,
        compute_training_metrics: bool = False,

        # Stopping Condition
        max_duration: Optional[Union[int, str, Time]] = None,

        # Algorithms
        algorithms: Optional[Union[Algorithm, Sequence[Algorithm]]] = None,

        # Optimizers and Scheduling
        optimizers: Optional[torch.optim.Optimizer] = None,
        schedulers: Optional[Union[ComposerScheduler, PyTorchScheduler, Sequence[Union[ComposerScheduler,
                                                                                       PyTorchScheduler]]]] = None,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        # Evaluators
        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        eval_interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,
        eval_subset_num_batches: int = -1,

        # Callbacks and Logging
        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,
        loggers: Optional[Union[LoggerDestination, Sequence[LoggerDestination]]] = None,
        run_name: Optional[str] = None,
        progress_bar: bool = True,
        log_to_console: Optional[bool] = None,
        console_log_level: Union[LogLevel, str, Callable[[State, LogLevel], bool]] = LogLevel.EPOCH,
        console_stream: Union[str, TextIO] = 'stderr',

        # Load Checkpoint
        load_path: Optional[str] = None,
        load_object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
        load_weights_only: bool = False,
        load_strict_model_weights: bool = False,
        load_progress_bar: bool = True,
        load_ignore_keys: Optional[Union[List[str], Callable[[Dict], None]]] = None,

        # Save Checkpoint
        save_folder: Optional[str] = None,
        save_filename: str = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        save_artifact_name: str = '{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}',
        save_latest_filename: Optional[str] = 'latest-rank{rank}.pt',
        save_latest_artifact_name: Optional[str] = '{run_name}/checkpoints/latest-rank{rank}',
        save_overwrite: bool = False,
        save_interval: Union[str, int, Time, Callable[[State, Event], bool]] = '1ep',
        save_weights_only: bool = False,
        save_num_checkpoints_to_keep: int = -1,

        # Graceful Resumption
        autoresume: bool = False,

        # DeepSpeed
        deepspeed_config: Optional[Dict[str, Any]] = None,

        # System/Numerics
        device: Optional[Union[str, Device]] = None,
        precision: Optional[Union[str, Precision]] = None,
        grad_accum: Union[int, str] = 1,

        # Reproducibility
        seed: Optional[int] = None,
        deterministic_mode: bool = False,

        # Distributed Training
        dist_timeout: float = 300.0,
        ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]] = None,

        # Grad Clip Norm
        grad_clip_norm: float = -1.0,

        # Profiling
        profiler: Optional[Profiler] = None,
    ):
        algorithms = list(ensure_tuple(algorithms))

        # Determine whether DeepSpeed is enabled
        deepspeed_enabled = deepspeed_config is not None

        # Device
        self._device = _get_device(device)

        # Distributed
        if deepspeed_enabled or dist.get_world_size() > 1:
            # deepspeed requires torch.distributed to be initialized, even if the world size is 1
            # distributed is always required with multi-rank training
            dist.initialize_dist(self._device, datetime.timedelta(seconds=dist_timeout))

        # Reproducibility
        rank_zero_seed, seed = _distribute_and_get_random_seed(seed, self._device)
        # If hparams is used to create the Trainer this function is called twice
        # which is okay because all runs with the hparams codepath will do this
        reproducibility.seed_all(seed)
        if deterministic_mode:
            reproducibility.configure_deterministic_mode()

        # Precision
        if precision is None:
            precision = Precision.AMP if isinstance(self._device, DeviceGPU) else Precision.FP32
        if isinstance(precision, str):
            precision = Precision(precision)

        _validate_precision(precision, self._device, deepspeed_enabled)

        # Optimizers and Schedulers
        if not optimizers:
            optimizers = DecoupledSGDW(list(model.parameters()), lr=0.1)
            # hard-coding the optimizer in the warning, as repr(optimizers) would print an annoying, multi-line warning
            warnings.warn(('No optimizer was specified. Defaulting to '
                           f"{type(optimizers).__name__}(lr={optimizers.defaults['lr']})"))

        num_optimizers = len(ensure_tuple(optimizers))
        if num_optimizers != 1:
            raise NotImplementedError(f'Only one optimizer is supported; found {num_optimizers} optimizers')

        # Move the model and optimizers to the device
        if not deepspeed_enabled:
            model = self._device.module_to_device(model)
            # Move any remaining optimizer parameters onto the device
            # It is possible that optimizer initialize created some internal tensors on CPU
            # that need to be moved onto GPU.
            optimizers = map_collection(optimizers, self._device.optimizer_to_device)

        # Grad Accum
        self.adaptive_gradient_accumulation = _is_adaptive_grad_accum(grad_accum, device=self._device)
        grad_accum = _get_initial_grad_accum(grad_accum)
        # Dynamic time estimate for forward and backward pass. Used for monitored_barrier to avoid deadlocks
        self.batch_compute_time = 300

        # Grad Clip Norm
        if grad_clip_norm > 0:

            warnings.warn(
                DeprecationWarning((f"Using the 'grad_clip_norm' field in Trainer is deprecated. Please use"
                                    'the GradientClipping Algorithm in composer.algorithms.gradient_clipping.')))

            if any(isinstance(alg, GradientClipping) for alg in algorithms):
                warnings.warn(
                    UserWarning(
                        f'The GradientClipping algorithm is already specified. Ignoring grad_clip_norm={grad_clip_norm}'
                    ))
            else:
                algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=grad_clip_norm))

        # Run Name
        if run_name is None:
            if autoresume:
                raise ValueError('When autoresume=True, the `run_name` must be specified.')
            run_name = _generate_run_name()
        log.info('Run name: %s', run_name)

        # Create the State
        self.state = State(rank_zero_seed=rank_zero_seed,
                           algorithms=algorithms,
                           model=model,
                           callbacks=callbacks,
                           grad_accum=grad_accum,
                           precision=precision,
                           optimizers=optimizers,
                           run_name=run_name,
                           deepspeed_config=deepspeed_config)

        # Profiler
        if profiler is not None:
            warnings.warn('The profiler is enabled. Using the profiler adds additional overhead when training.')
            self.state.profiler = profiler
            self.state.profiler.bind_to_state(self.state)

        # Console Logging
        loggers = list(ensure_tuple(loggers))
        if any(isinstance(x, ProgressBarLogger) for x in loggers):
            warnings.warn(
                DeprecationWarning(
                    (f'Specifying the {ProgressBarLogger.__name__} via `loggers` is deprecated. Instead, '
                     'please specify `progress_bar`, `log_to_console`, `log_level`, and `stream` arguments when '
                     'constructing the trainer. If specified, these arguments will be ignored, as the '
                     f'{ProgressBarLogger.__name__} was already created.')))
        else:
            loggers.append(
                ProgressBarLogger(
                    progress_bar=progress_bar,
                    log_to_console=log_to_console,
                    console_log_level=console_log_level,
                    stream=console_stream,
                ))

        # Logger
        self.logger = Logger(state=self.state, destinations=loggers)

        # Callbacks
        self.state.callbacks[:] = list(cast(List[Callback], loggers)) + self.state.callbacks

        # Checkpoint Saving
        self._checkpoint_saver = None
        if save_folder is not None:
            self._checkpoint_saver = CheckpointSaver(
                folder=save_folder,
                filename=save_filename,
                artifact_name=save_artifact_name,
                latest_filename=save_latest_filename,
                latest_artifact_name=save_latest_artifact_name,
                overwrite=save_overwrite,
                weights_only=save_weights_only,
                save_interval=save_interval,
                num_checkpoints_to_keep=save_num_checkpoints_to_keep,
            )
            self.state.callbacks.append(self._checkpoint_saver)

        # The Engine
        self.engine = Engine(state=self.state, logger=self.logger)

        # Set the logger
        model.logger = self.logger

        # Run Event.INIT
        self.engine.run_event(Event.INIT)

        # After running Event.INIT, then set the "optional" elements of state that could be passed in on FIT instead of INIT
        # Setting these attributes here ensures that algorithms do not depend on unavailable attributes during Event.INIT

        # Train Dataloader
        self._train_data_spec = None if train_dataloader is None else ensure_data_spec(train_dataloader)
        if self._train_data_spec is not None:
            self.state.set_dataloader(self._train_data_spec.dataloader, train_dataloader_label,
                                      train_subset_num_batches)
            self.state.train_dataloader = self.state.dataloader
        self.train_metrics = _get_training_metrics(model) if compute_training_metrics else None

        # Max Duration
        if max_duration is not None:
            self.state.max_duration = ensure_time(max_duration, TimeUnit.EPOCH)

        self.logger.data_fit({'rank_zero_seed': rank_zero_seed})

        assert isinstance(self.state.model, ComposerModel)
        self._original_model = self.state.model

        # Schedulers
        self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)
        if scale_schedule_ratio != 1.0:
            if len(self.state.schedulers) == 0:
                raise ValueError('Specifying `scale_schedule_ratio` without `schedulers` has no effect.')
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)

        if step_schedulers_every_batch is None:
            self._scheduler_step_frequency = _get_default_scheduler_frequency(schedulers)
        else:
            self._scheduler_step_frequency = TimeUnit.BATCH if step_schedulers_every_batch else TimeUnit.EPOCH

        # Evaluators
        if eval_dataloader is None:
            evaluators: List[Evaluator] = []
        else:
            evaluators = [
                ensure_evaluator(evaluator, model.metrics(train=False)) for evaluator in ensure_tuple(eval_dataloader)
            ]
            _set_evaluator_interval_and_subset_num_batches(
                evaluators=evaluators,
                eval_interval=eval_interval,
                subset_num_batches=eval_subset_num_batches,
            )
        if len(evaluators) == 0:
            if eval_subset_num_batches != -1:
                raise ValueError('Specifying `eval_subset_num_batches` without an `eval_dataloader` has no effect.')
            if eval_interval != 1:
                raise ValueError('Specifying `eval_interval` without an `eval_dataloader` has no effect.')
        self.state.evaluators = evaluators

        # Some algorithms require specific settings
        self._backwards_create_graph = any(map(lambda x: x.backwards_create_graph, ensure_tuple(algorithms)))
        self._find_unused_parameters = any(map(lambda x: x.find_unused_parameters, ensure_tuple(algorithms)))
        self._ddp_sync_strategy = _get_ddp_sync_strategy(ddp_sync_strategy, self._find_unused_parameters)

        # Configure Deepspeed
        if self.state.deepspeed_config is not None:
            try:
                import deepspeed
            except ImportError as e:
                raise MissingConditionalImportError(
                    extra_deps_group='deepspeed',
                    conda_package='deepspeed>=0.5.5',
                    conda_channel=None,
                ) from e
            self.state.deepspeed_config = _parse_deepspeed_config(self.state.deepspeed_config, state=self.state)
            optimizer = ensure_tuple(self.state.optimizers)[0]
            log.debug('Initializing deepspeed')
            (self.state.model, self.state.optimizers, _, _) = deepspeed.initialize(config=self.state.deepspeed_config,
                                                                                   model=self.state.model,
                                                                                   optimizer=optimizer)
            # Since the DeepSpeed ZeRO optimizer does not inherit torch.optim.Optimizer, the schedulers must be
            # compiled and bound BEFORE DeepSpeed initialization. However, this is OK, as the the DeepSpeed Zero
            # optimizer uses the same underlying parameter groups as the original optimizer. See
            # * https://github.com/microsoft/DeepSpeed/blob/fee73135980e78f8be7e1a3ff556751623ef6aaa/deepspeed/runtime/zero/stage_1_and_2.py#L1911-L1917
            # * https://github.com/microsoft/DeepSpeed/blob/ef17c89570ceae5b26a5f886e9d8cd0941afc0ac/deepspeed/runtime/zero/stage3.py#L2532-L2538
            # In addition, the deepspeed engine is responsible for serializing the model and optimizer state,
            # so these attributes should not be serialized with the composer state.
            if 'model' in self.state.serialized_attributes:
                self.state.serialized_attributes.remove('model')

            if 'optimizers' in self.state.serialized_attributes:
                self.state.serialized_attributes.remove('optimizers')

        # If using DeepSpeed, the model must be loaded from checkpoint after the engine has been
        # initialized, but if using PyTorch DDP, the model must be loaded before it is wrapped with
        # DDP.

        # Load Checkpoint
        self._rng_state = None
        # If autoresume is enabled, first check for existing checkpoints to load
        if autoresume:
            log.info('Searching for a previous checkpoint to autoresume')
            if save_folder is None:
                raise ValueError('The `save_folder` must be specified when autoresume is enabled.')
            if save_overwrite:
                raise ValueError(
                    'The flag `save_overwrite` must be False when autoresume is enabled as autoresume always loads the '
                    'latest existing checkpoint in `save_folder`.')
            if save_latest_filename is None:
                raise ValueError(
                    'The `save_latest_filename` must be specified so autoresume knows where to load checkpoints from.')
            if save_latest_artifact_name is None:
                raise ValueError(
                    'The `save_latest_artifact_name` must be specified so autoresume can load the latest checkpoint.')
            if run_name is None:
                raise ValueError(
                    'The `run_name` must be specified when using autoresume so Event.INIT is run with the correct run name.'
                )
            autoresume_checkpoint_path = self._get_autoresume_checkpoint(
                save_folder=save_folder,
                save_latest_filename=save_latest_filename,
                save_latest_artifact_name=save_latest_artifact_name,
                loggers=loggers,
                load_progress_bar=load_progress_bar)
            # Found latest checkpoint path, load that instead
            if autoresume_checkpoint_path:
                load_path = autoresume_checkpoint_path
                # Disable object_store since _get_autoresume_checkpoint will download the checkpoint
                # To the save folder, if needed.
                load_object_store = None
                # Disable `load_weights_only` since this applies only to the initial training run
                load_weights_only = False
                log.info('Autoresuming training from checkpoint')
            else:
                log.info('No previous autoresume checkpoint found')
        # Actually load the checkpoint from potentially updated arguments
        if load_path is not None:
            self._rng_state = load_checkpoint(
                state=self.state,
                path=load_path,
                object_store=load_object_store,
                load_weights_only=load_weights_only,
                strict_model_weights=load_strict_model_weights,
                progress_bar=load_progress_bar,
                ignore_keys=load_ignore_keys,
            )
            self.state.run_name = run_name

        # reseed here. This helps with a couple of issues:
        # 1. rng state may change at Event.INIT. For example, if an algorithm creates a new module and module
        # parameters are initialized randomly, rng state will change. This reseeding nullifies such effects.
        # 2. While resuming from a checkpoint, we want to spin dataloader and bring it back to the same state as at the time
        # of the checkpoint. Therefore, spinning needs to start from the same rng state as in the original run.
        log.info(f'Setting seed to {self.state.seed}')
        reproducibility.seed_all(self.state.seed)

        # Move the model and optimizers to the specified device
        if not self.deepspeed_enabled and dist.get_world_size() > 1:
            # Only wrap the module if required
            self.state.model = prepare_ddp_module(self.state.model, self._find_unused_parameters)

    @property
    def deepspeed_enabled(self):
        """Whether DeepSpeed is enabled.

        .. seealso:: `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_
        """
        return is_model_deepspeed(self.state.model)

    @property
    def saved_checkpoints(self) -> List[Tuple[Timestamp, List[pathlib.Path]]]:
        """The checkpoint timestamps and filepaths.

        This list contains tuples of the save timestamp and the checkpoint filepaths.
        This list will have at most ``save_num_checkpoints_to_keep`` entries. The latest checkpoint
        will be at the end.

        .. note::

            When using DeepSpeed, the index of a filepath in each list corresponds to the global rank of
            the process that wrote that file. Each filepath is valid only on the process's (rank's) node.

            Otherwise, when not using DeepSpeed, each sub-list will contain only one filepath since only rank zero
            saves checkpoints.
        """
        if self._checkpoint_saver is None:
            return []
        return self._checkpoint_saver.saved_checkpoints

    def _get_autoresume_checkpoint(
        self,
        save_folder: str,
        save_latest_filename: str,
        save_latest_artifact_name: str,
        loggers: Sequence[LoggerDestination],
        load_progress_bar: bool,
    ):
        """Determines the load path when using autoresume.

        First, check the ``save_folder`` for the latest checkpoint.
        If no latest checkpoint is found locally, then check each logger for the latest checkpoint, and download
        it to the ``save_folder``.

        Returns:
            Optional[str]: The path to the latest checkpoint, if found, otherwise None.
        """
        save_latest_filename = format_name_with_dist(save_latest_filename, self.state.run_name)
        save_folder = format_name_with_dist(save_folder, self.state.run_name)
        save_latest_artifact_name = format_name_with_dist(save_latest_artifact_name, self.state.run_name)
        latest_checkpoint_path = os.path.join(save_folder, save_latest_filename)
        # If latest checkpoint is not saved locally, try to fetch from loggers
        if not os.path.exists(latest_checkpoint_path):
            # Make save folder in case it doesn't exist so latest checkpoint can be downloaded
            os.makedirs(save_folder, exist_ok=True)
            for logger in loggers:
                try:
                    # Fetch from logger. If it succeeds, stop trying the rest of the loggers
                    get_file(
                        path=save_latest_artifact_name,
                        destination=latest_checkpoint_path,
                        object_store=logger,
                        overwrite=True,
                        progress_bar=load_progress_bar,
                    )
                    break
                except (NotImplementedError, FileNotFoundError):
                    # Ignore errors caused by no checkpoint saved with logger
                    pass
        # Require all ranks to have local checkpoint if we wish to restore from it
        latest_checkpoint_exists = self._device.tensor_to_device(
            torch.tensor([os.path.exists(latest_checkpoint_path)], dtype=torch.uint8))
        dist.all_reduce(latest_checkpoint_exists, reduce_operation='MIN')
        # If latest checkpoint is saved locally, change load_path to it
        if int(latest_checkpoint_exists.item()) == 1:
            return latest_checkpoint_path

    def fit(
        self,
        *,
        # Train Dataloader
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = 'train',
        train_subset_num_batches: Optional[int] = None,
        compute_training_metrics: Optional[bool] = None,

        # Timing
        duration: Optional[Union[int, str, Time[int]]] = None,
        reset_time: bool = False,

        # Schedulers
        schedulers: Optional[Union[ComposerScheduler, PyTorchScheduler, Sequence[Union[ComposerScheduler,
                                                                                       PyTorchScheduler]]]] = None,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        # Evaluation
        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        eval_subset_num_batches: int = -1,
        eval_interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,

        # Numerics
        grad_accum: Optional[Union[int, str]] = None,
        precision: Optional[Union[str, Precision]] = None,
    ):
        """Train the model.

        The Composer :class:`.Trainer` supports multiple calls to :meth:`.fit`. Any arguments specified during
        the call to :meth:`.fit` will override the values specified when constructing the :class:`.Trainer`.
        All arguments are optional, with the following exceptions:

        *   The ``train_dataloader`` must be specified here if not provided when constructing the :class:`.Trainer`.
        *   The ``duration`` must be specified here if not provided when constructing the :class:`.Trainer`,
            or if this is a subsequent call to :meth:`.fit`.

        For example, the following are equivalent:

        .. testcode::

            # The `train_dataloader` and `duration` can be specified
            # when constructing the Trainer
            trainer_1 = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                max_duration="1ep",
            )
            trainer_1.fit()

            # Or, these arguments can be specified on `fit()`
            trainer_2 = Trainer(model)
            trainer_2.fit(
                train_dataloader=train_dataloader,
                duration="1ep"
            )

        When invoking :meth:`.fit` for a subsequent time, either ``reset_time`` or ``duration`` must be specified.
        Otherwise, it is ambiguous for how long to train.

        *   If ``reset_time`` is True, then :meth:`.fit` will train for the same amount of time as the previous
            call (or for ``duration`` if that parameter is also specified). The :attr:`.State.timestamp` will be reset,
            causing :class:`.ComposerScheduler` and :class:`.Algorithm` instances to start from the beginning, as if it
            is a new training run. Model gradients, optimizer states, and native PyTorch schedulers will not be reset.

        *   If ``reset_time`` is False, then :meth:`.fit` will train for the amount of time specified by
            ``duration``. The :attr:`.State.max_duration` will be incremented by ``duration``.

        For example:

        .. testcode::

            # Construct the trainer
            trainer = Trainer(max_duration="1ep")

            # Train for 1 epoch
            trainer.fit()
            assert trainer.state.timestamp.epoch == "1ep"

            # Reset the time to 0, then train for 1 epoch
            trainer.fit(reset_time=True)
            assert trainer.state.timestamp.epoch == "1ep"

            # Train for another epoch (2 epochs total)
            trainer.fit(duration="1ep")
            assert trainer.state.timestamp.epoch == "2ep"

            # Train for another batch (2 epochs + 1 batch total)
            # It's OK to switch time units!
            trainer.fit(duration="1ba")
            assert trainer.state.timestamp.epoch == "2ep"
            assert trainer.state.timestamp.batch_in_epoch == "1ba"

            # Reset the time, then train for 3 epochs
            trainer.fit(reset_time=True, duration="3ep")
            assert trainer.state.timestamp.epoch == "3ep"

        Args:
            train_dataloader (Iterable | DataSpec | Dict[str, Any], optional): See :class:`.Trainer`.
            train_dataloader_label (str, optional): See :class:`.Trainer`.
            train_subset_num_batches (int, optional): See :class:`.Trainer`.
            compute_training_metrics (bool, optional): See :class:`.Trainer`.
            reset_time (bool): Whether to reset the :attr:`.State.timestamp` to zero values. Defaults to False.

                If ``True``, the timestamp will be zeroed out, causing :class:`.ComposerScheduler` and
                :class:`.Algorithm` instances to start from the beginning, as if it is a new training run. The model
                will be trained for ``duration``, if specified, or for :attr:`.State.max_duration`, which would have
                been provided when constructing the :class:`.Trainer` or by a previous call to :meth:`.fit`.

                .. note::

                    Model gradients, optimizer states, and native PyTorch schedulers will not be reset.

                If ``False`` (the default), training time will be incremented from where the previous call to
                :meth:`.fit` finished (or from zero, if a new training run).
                The :attr:`~.State.max_duration` will be incremented by the ``duration`` parameter.

            duration (Time[int] | str | int, optional): The duration to train. Can be an integer, which will be
                interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), or a :class:`.Time` object.

                If ``reset_time`` is False (the default), then :attr:`.State.max_duration` will be converted
                into the same units as this parameter (if necessary), and then the max duration incremented by the
                value of this parameter.

                If ``reset_time`` is True, then :attr:`.State.max_duration` will be set to this parameter.

            optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): See :class:`.Trainer`.
            schedulers (PyTorchScheduler | ComposerScheduler | Sequence[PyTorchScheduler | ComposerScheduler], optional): See :class:`.Trainer`.
            scale_schedule_ratio (float, optional): See :class:`.Trainer`.
            step_schedulers_every_batch (bool, optional): See :class:`.Trainer`.
            eval_dataloader (Iterable | DataSpec | Evaluator | Sequence[Evaluator], optional): See :class:`.Trainer`.
            eval_subset_num_batches (int, optional): See :class:`.Trainer`.
            eval_interval (int | str | Time | (State, Event) -> bool, optional): See :class:`.Trainer`.
            grad_accum (int | str, optional): See :class:`.Trainer`.
            precision (Precision | str, optional): See :class:`.Trainer`.
        """
        # Train Dataloader
        if train_dataloader is not None:
            self._train_data_spec = ensure_data_spec(train_dataloader)
            self.state.set_dataloader(self._train_data_spec.dataloader, train_dataloader_label)
            self.state.train_dataloader = self.state.dataloader
        if self._train_data_spec is None:
            _raise_missing_argument_exception('train_dataloader')
        if train_subset_num_batches is not None:
            self.state.dataloader_len = train_subset_num_batches
        if compute_training_metrics is not None:
            self.train_metrics = _get_training_metrics(self._original_model) if compute_training_metrics else None

        # Reset Time
        if reset_time:
            self.state.timestamp = Timestamp()

        # Max Duration
        if duration is not None:
            duration = ensure_time(duration, TimeUnit.EPOCH)
            # Effectively increment the max duration (if not resetting the Time)
            # or set the max_duration (if resetting the time -- self.state.timestamp.get(duration.unit) will be 0)
            # It is important to set the duration, rather than incrementing it, as ``duration`` could be in
            # different units than ``max_duration``
            self.state.max_duration = duration + self.state.timestamp.get(duration.unit)

        if self.state.max_duration is None:
            _raise_missing_argument_exception('max_duration')

        if self.state.max_duration <= self.state.timestamp.get(self.state.max_duration.unit) and not reset_time:
            raise ValueError(
                (f'The max_duration ({self.state.max_duration}) is less than or equal to the elapsed training duration '
                 f'({self.state.timestamp.get(self.state.max_duration.unit)}). No training would occur. '
                 'Please provide the `duration` or specify `reset_time=True` in Trainer.fit().'))

        # Scale Schedule Ratio and Schedulers
        if scale_schedule_ratio != 1.0:
            # Not scaling the schedulers if the ratio is 1.0 in case if the scheduler cannot be scaled
            # (e.g. a custom LambdaLR). However, since 1.0 implies no scaling, it is still possible
            # to train with it.
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)
        if schedulers is not None:
            self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)

            if step_schedulers_every_batch is None:
                self._scheduler_step_frequency = _get_default_scheduler_frequency(schedulers)
            else:
                self._scheduler_step_frequency = TimeUnit.BATCH if step_schedulers_every_batch else TimeUnit.EPOCH
        else:
            if scale_schedule_ratio != 1.0:
                raise ValueError('Specifying `scale_schedule_ratio` without `schedulers` has no effect.')

            if step_schedulers_every_batch is not None:
                raise ValueError('Specifying `step_schedulers_every_batch` without `schedulers` has no effect.')

            if step_schedulers_every_batch is not None:
                raise ValueError('Specifying `step_schedulers_every_batch` without `schedulers` has no effect.')

        # Evaluators
        if eval_dataloader is not None:
            evaluators = [
                # Need to use the `original_model` rather than `state.model`, as `state.model`
                # could be DDP / DeepSpeed wrapped.
                ensure_evaluator(evaluator, default_metrics=self._original_model.metrics(train=False))
                for evaluator in ensure_tuple(eval_dataloader)
            ]
            _set_evaluator_interval_and_subset_num_batches(
                evaluators=evaluators,
                eval_interval=eval_interval,
                subset_num_batches=eval_subset_num_batches,
            )
            if len(evaluators) == 0:
                if eval_subset_num_batches != -1:
                    raise ValueError('Specifying `eval_subset_num_batches` without an `eval_dataloader` has no effect.')
                if eval_interval != 1:
                    raise ValueError('Specifying `eval_interval` without an `eval_dataloader` has no effect.')

            self.state.evaluators = evaluators

        # Grad Accum
        if grad_accum is not None:
            self.adaptive_gradient_accumulation = _is_adaptive_grad_accum(grad_accum, device=self._device)
            self.state.grad_accum = _get_initial_grad_accum(grad_accum)

        # Precision
        if precision is not None:
            if self.deepspeed_enabled:
                raise ValueError('Changing the precision when using DeepSpeed is not supported')
            precision = Precision(precision)
            _validate_precision(precision, self._device, self.deepspeed_enabled)
            self.state.precision = precision

        self._train_loop()

    def close(self):
        """Shutdown the trainer.

        .. seealso:: :meth:`.Engine.close` for additional information.
        """
        self.engine.close()

    def _ensure_metrics_device_and_dtype(self, metrics: MetricCollection):
        # Safety check to ensure the metric and data are on the same device. Normally not
        # needed because the metric is automatically on the same device as the model.
        # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html for details.
        metrics = self._device.module_to_device(metrics)

        # HACK: DeepSpeed somehow manages to convert metric internal states to its own dtype. When
        # running with FP16, this tends to result in overflows. Let's assume FP32 is good enough.
        for _, metric in metrics.items():
            metric.set_dtype(torch.float32)  # type: ignore

        return metrics

    def _compute_and_log_metrics(self, dataloader_label: str, log_level: LogLevel, metrics: MetricCollection):
        """Computes metrics, logs the results, and updates the state.

        Args:
            dataloader_label (str): The dataloader label.
            metrics (MetricCollection): The metrics to compute.
            log_level (LogLevel): The LogLevel for logging metrics.
        """
        computed_metrics = metrics.compute()
        self.logger.data(
            log_level=log_level,
            data={f'metrics/{dataloader_label}/{name}': val for (name, val) in computed_metrics.items()},
        )
        self.state.current_metrics[dataloader_label] = computed_metrics

    def _spin_dataloaders(self):
        """Spin the dataloaders to restore sampler state.

        Only one batch must be loaded to seed the sampler's generator. since only the first batch is being loaded, the
        dataloader may not be completely iterated through.
        """
        # spin the evaluator dataloaders once to initialize its sampler deterministically
        # so it does not affect any other RNG reads
        log.debug('Spinning the dataloaders')
        for evaluator in self.state.evaluators:
            dataloader = evaluator.dataloader.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(0)
            for _ in dataloader:
                break

        # spin the train dataloader's sampler to get to the state of the desired epoch
        dataloader = self.state.dataloader
        assert dataloader is not None, 'train dataloader is set on state after FIT_START'
        for epoch in range(int(self.state.timestamp.epoch)):
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
            for _ in dataloader:
                break

    def _accumulate_time_across_ranks(
        self,
        num_samples: int,
        num_tokens: int,
        batch_time: datetime.timedelta,
    ) -> Tuple[int, int, datetime.timedelta]:
        """Accumulate the number of samples and tokens across ranks.

        Returns a (num_samples, num_tokens, batch_time) tuple.
        """
        # Samples and tokens should be summed
        # Batch time should be the value from rank 0
        sample_token_tensor = self._device.tensor_to_device(torch.tensor([num_samples, num_tokens], dtype=torch.int))
        dist.all_reduce(sample_token_tensor, reduce_operation='SUM')

        batch_time_tensor = self._device.tensor_to_device(
            torch.tensor([batch_time.total_seconds()], dtype=torch.float64))
        dist.broadcast(batch_time_tensor, src=0)
        batch_time = datetime.timedelta(seconds=batch_time_tensor[0].cpu().item())

        return int(sample_token_tensor[0].cpu().item()), int(sample_token_tensor[1].cpu().item()), batch_time

    def _train_loop(self) -> None:
        """Run training for the specified number of epochs and log results."""
        # print training start
        log.info('Using precision %s', self.state.precision)
        self.logger.data_fit({algo.__class__.__name__: 1 for algo in self.state.algorithms})

        assert self.state.dataloader is not None, 'dataloader is set in __init__() or fit()'
        assert self._train_data_spec is not None, 'The train data spec is set in __init__() or fit()'

        self.engine.run_event(Event.FIT_START)

        # surpressing GradScaler warnings as they are always created
        # self._use_grad_scaling() will raise a RuntimeError if grad scaling is not available when it is required
        warnings.filterwarnings(action='ignore', message='torch.cuda.amp.GradScaler')
        self.state.scaler = ClosureGradScaler() if self._use_closures() else GradScaler()
        use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

        self._spin_dataloaders()

        if self.state.timestamp.batch_in_epoch == 0 and self._rng_state is not None:
            # only restore the rng state here if the step in the current epoch is zero.
            reproducibility.load_rng_state(self._rng_state)
            self._rng_state = None

        if self.train_metrics is not None:
            self.train_metrics = self._ensure_metrics_device_and_dtype(self.train_metrics)

        # Flag if the epoch finished early, so it can be tracked whether to run the epoch end events
        finished_epoch_early = False

        last_wct = datetime.datetime.now()

        while self.state.timestamp < self.state.max_duration:
            self.state.model.train()

            if int(self.state.timestamp.batch_in_epoch) == 0:
                self.engine.run_event(Event.EPOCH_START)
                self.logger.data_epoch({'epoch': int(self.state.timestamp.epoch)})
                if self.train_metrics is not None:
                    # reset the metrics before every epoch
                    self.train_metrics.reset()

            dataloader = self.state.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(int(self.state.timestamp.epoch))

            for batch_idx, self.state.batch in enumerate(self._iter_dataloader()):

                # if resuming, skip dataloader forward to the minibatch index
                if batch_idx < int(self.state.timestamp.batch_in_epoch):
                    # Restore the RNG state immediately before the next batch is yielded from the dataloader
                    if batch_idx + 1 == int(self.state.timestamp.batch_in_epoch) and self._rng_state is not None:
                        reproducibility.load_rng_state(self._rng_state)
                        self._rng_state = None
                    continue

                self.state.batch = self._device.batch_to_device(self.state.batch)
                self.state.batch = self._train_data_spec.device_transforms(self.state.batch)
                rank_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
                rank_num_tokens = self._train_data_spec.get_num_tokens_in_batch(self.state.batch)

                if self.deepspeed_enabled:
                    self.state.batch = _fix_batch_precision_for_deepspeed(self.state.batch, self.state.precision)

                if self.train_metrics is not None:
                    self.state.model.eval()
                    with torch.no_grad():
                        for eval_microbatch in self._train_data_spec.split_batch(self.state.batch,
                                                                                 self.state.grad_accum):
                            # TODO: Detect if self.run_event(Event.AFTER_DATALOADER) changes the training
                            # data and if so print a warning that metrics may return unexpected results
                            with get_precision_context(self.state.precision):
                                outputs, targets = self._original_model.validate(eval_microbatch)
                                # Run in same precision context to avoid NaNs
                                self.train_metrics.update(outputs, targets)

                self.state.model.train()

                self.engine.run_event(Event.AFTER_DATALOADER)

                self.engine.run_event(Event.BATCH_START)
                self.logger.data_batch({
                    'trainer/global_step': int(self.state.timestamp.batch),
                    'trainer/batch_idx': self.state.timestamp.batch_in_epoch.value,
                })

                total_loss = self._train_batch(use_grad_scaling)

                if use_grad_scaling:
                    self.state.scaler.update()

                if total_loss is not None:
                    if not isinstance(total_loss, torch.Tensor):
                        total_loss = self._device.tensor_to_device(torch.tensor([total_loss]))

                    # total_loss can be None if gradient scaling failed
                    dist.all_reduce(total_loss, reduce_operation='SUM')
                    full_loss = total_loss.cpu().item()
                    self.logger.data_batch({'loss/train': full_loss / dist.get_world_size()})

                # The scheduler step.step() and compute_and_log_metrics() are going to be included in the
                # next batch's wall clock time. The time accumulation must be done here so schedulers
                # have the latest timing information

                now = datetime.datetime.now()

                batch_time = now - last_wct

                total_num_samples, total_num_tokens, batch_time = self._accumulate_time_across_ranks(
                    rank_num_samples,
                    rank_num_tokens,
                    batch_time,
                )

                # `now` is actually in the past, but want to include the time it takes to perform this reduction
                last_wct = now

                self.state.timestamp = self.state.timestamp.to_next_batch(
                    samples=total_num_samples,
                    tokens=total_num_tokens,
                    duration=batch_time,
                )

                if self._scheduler_step_frequency == TimeUnit.BATCH:
                    for scheduler in self.state.schedulers:
                        scheduler.step()

                if self.train_metrics is not None:
                    self._compute_and_log_metrics(
                        dataloader_label='train',
                        log_level=LogLevel.BATCH,
                        metrics=self.train_metrics,
                    )

                self.engine.run_event(Event.BATCH_END)

                # Pause the timing during evaluation
                # Evaluation time is tracked separately in state.eval_timestamp
                duration = datetime.datetime.now() - last_wct
                self._run_evaluators(Event.BATCH_END, log_level=LogLevel.BATCH)
                last_wct = datetime.datetime.now() - duration

                self.engine.run_event(Event.BATCH_CHECKPOINT)

                if self.state.timestamp >= self.state.max_duration:
                    # If max_duration is specified in batches, samples, or tokens, and
                    # and the max_duration is reached mid-epoch, then break out of the dataloader
                    # to finish the epoch early and finish training.
                    finished_epoch_early = True
                    break

            if not finished_epoch_early or self.state.dataloader_len == self.state.timestamp.batch_in_epoch:
                # Trigger the epoch end events if the dataloader was exhausted.
                # This happens if the "break" did not trigger above, or if it
                # did (e.g. duration specified in samples/batches/tokens), but it is still
                # the end of the dataloader (i.e. next(dataloader) would raise StopIteration)
                self.state.timestamp = self.state.timestamp.to_next_epoch()

                if self.train_metrics is not None:
                    self._compute_and_log_metrics(
                        dataloader_label='train',
                        log_level=LogLevel.EPOCH,
                        metrics=self.train_metrics,
                    )

                if self._scheduler_step_frequency == TimeUnit.EPOCH:
                    for scheduler in self.state.schedulers:
                        scheduler.step()

                self.engine.run_event(Event.EPOCH_END)

                # Pause the timing during evaluation
                # Evaluation time is tracked separately in state.eval_timestamp
                duration = datetime.datetime.now() - last_wct
                self._run_evaluators(Event.EPOCH_END, log_level=LogLevel.EPOCH)
                last_wct = datetime.datetime.now() - duration

                self.engine.run_event(Event.EPOCH_CHECKPOINT)
        self.engine.run_event(Event.FIT_END)
        self._run_evaluators(Event.FIT_END, log_level=LogLevel.FIT)

    def _run_evaluators(self, event: Event, log_level: LogLevel):
        """Runs evaluators periodically during training."""
        for evaluator in self.state.evaluators:
            assert evaluator.eval_interval is not None, 'eval_interval should have been set on __init__() or fit()'
            assert evaluator.subset_num_batches is not None, 'subset_num_batches should have been set on __init__() or fit()'
            if evaluator.eval_interval(self.state, event):
                self.eval(
                    dataloader=evaluator.dataloader,
                    dataloader_label=evaluator.label,
                    subset_num_batches=evaluator.subset_num_batches,
                    metrics=evaluator.metrics,
                    log_level=log_level,
                )

    def _train_batch(self, use_grad_scaling: bool):
        """Compute loss by training on a full batch of data.

        Adaptively change microbatch size if enabled to maximize GPU usage.

        Args:
            use_grad_scaling (bool): Enables gradient scaling
        """
        assert self._train_data_spec is not None, 'The train data spec should be set on __init__ or fit()'

        # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop
        # TODO: fix this name collision!
        device_batch = self.state.batch

        # Retry until we successfully complete training and return loss
        while True:
            start_time = time.time()
            total_loss = None
            # Note: We use uint8 instead of bool as BOR is not supported on all torch.distributed backends
            should_handle_cuda_oom = 0
            try:
                assert self.state.scaler is not None
                microbatches = self._train_data_spec.split_batch(device_batch, self.state.grad_accum)
                if self.deepspeed_enabled:
                    total_loss = self._train_microbatches(microbatches)
                elif self._use_closures():
                    for optimizer in self.state.optimizers:
                        if use_grad_scaling:
                            total_loss = self.state.scaler.step(
                                optimizer, closure=lambda **kwargs: self._train_microbatches(microbatches, **kwargs))
                        else:
                            total_loss = optimizer.step(
                                closure=lambda **kwargs: self._train_microbatches(microbatches, **kwargs).item())
                else:
                    total_loss = self._train_microbatches(microbatches)
                    for optimizer in self.state.optimizers:
                        if use_grad_scaling:
                            self.state.scaler.step(optimizer)
                        else:
                            optimizer.step()
            except RuntimeError as e:
                if self.adaptive_gradient_accumulation and _is_cuda_oom(e):
                    log.debug((f"Rank {dist.get_global_rank()} OOM'd."))
                    should_handle_cuda_oom = 1
                else:
                    raise
            end_time = time.time()

            # Use monitored barrier to error on deadlock. If one rank OOMs and another doesn't and gets stuck
            # on a dist reduction in gradient syncronization, the monitored barrier will fail after the timeout.
            # If `adaptive_gradient_accumulation=False`, the OOMing rank will instead crash, avoiding deadlock risk.
            if self.adaptive_gradient_accumulation:
                try:
                    dist.monitored_barrier(timeout=datetime.timedelta(seconds=max(10, 0.5 * self.batch_compute_time)))
                except RuntimeError as e:
                    raise RuntimeError(
                        'A deadlock was encountered in the train loop, likely because a strict subset of '
                        'ranks encountered CUDA OOM when `grad_accum=auto`. Try manually setting `grad_accum` '
                        'instead.') from e

            # Propagate across all ranks if any rank hit CUDA OOM
            should_handle_cuda_oom = self._device.tensor_to_device(
                torch.tensor([should_handle_cuda_oom], dtype=torch.uint8))
            dist.all_reduce(should_handle_cuda_oom, reduce_operation='MAX')
            # Check if any rank hit CUDA OOM
            if int(should_handle_cuda_oom.item()) == 1:
                # If any rank hit CUDA OOM, update grad_accum and retry. Ignore any caught_timeout_error since
                # it is likely transient, e.g. timeout because certain ranks OOMed and didn't reach barrier.
                # Raise runtime error if training 1 sample at a time still resulted in CUDA out of memory
                device_batch_size = self._train_data_spec.get_num_samples_in_batch(device_batch)
                if self.state.grad_accum == device_batch_size:
                    raise RuntimeError(
                        ('CUDA out of memory. The train loop failed with an internal microbatch of size 1.'
                         'The GPU does not have enough memory to process even 1 sample.'))
                else:
                    original_grad_accum = self.state.grad_accum
                    self.state.grad_accum = min(2 * self.state.grad_accum, device_batch_size)
                    warnings.warn(
                        RuntimeWarning('CUDA out of memory detected. Gradient Accumulation '
                                       f'increased from {original_grad_accum} -> {self.state.grad_accum}, '
                                       'and the batch will be retrained.'))
            # Otherwise, log grad_accum and return calculated loss
            else:
                # Synchronize new batch compute time
                batch_compute_time = end_time - start_time
                batch_compute_time = self._device.tensor_to_device(torch.tensor([batch_compute_time],
                                                                                dtype=torch.float))
                dist.all_reduce(batch_compute_time, reduce_operation='MAX')
                self.batch_compute_time = batch_compute_time.item()

                self.logger.data_batch({'trainer/grad_accum': self.state.grad_accum})
                return total_loss

    def _train_microbatches(self, microbatches: Sequence[Batch], ddp_sync: bool = True):
        """Iterate over microbatches and compute the loss that will be used to step the optimizer.

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

        assert self._train_data_spec is not None

        with context():
            self.engine.run_event(Event.BEFORE_TRAIN_BATCH)

            assert self.state.optimizers is not None
            assert self.state.scaler is not None

            use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

            if not self.deepspeed_enabled:
                for optimizer in self.state.optimizers:
                    optimizer.zero_grad()

            # tracker for gradient accumulation
            total_loss = self._device.tensor_to_device(torch.zeros(size=(1,)))
            current_batch_size = sum([self._train_data_spec.get_num_samples_in_batch(batch) for batch in microbatches])

            for microbatch_idx, self.state.batch in enumerate(microbatches):
                is_final_microbatch = microbatch_idx + 1 == len(microbatches)
                self._train_microbatch(use_grad_scaling, current_batch_size, total_loss, is_final_microbatch)

            # Unscale gradients before `Event.AFTER_TRAIN_BATCH`
            if use_grad_scaling:
                for optimizer in ensure_tuple(self.state.optimizers):
                    self.state.scaler.unscale_(optimizer)

            self.engine.run_event(Event.AFTER_TRAIN_BATCH)

            return total_loss

    def _train_microbatch(self, use_grad_scaling: bool, current_batch_size: int, total_loss: torch.Tensor,
                          is_final_microbatch: bool):
        """Train and compute the loss of ``state.batch``, which is assumed to be a single microbatch.

        Args:
            use_grad_scaling (bool): Whether to use gradient scaling.
            current_batch_size (int): The current batch size.
            minibatch_num_samples (int): Number of samples in the minibatch.
            total_loss (torch.Tensor): Total loss aggregated across all microbatches.
            is_final_microbatch (bool): If current microbatch is the last one.
        """
        assert self.state.scaler is not None
        assert self._train_data_spec is not None

        microbatch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
        sync_context = contextlib.nullcontext() if self.deepspeed_enabled else ddp_sync_context(
            self.state,
            is_final_microbatch,
            self._ddp_sync_strategy,
        )

        with sync_context:
            # forward pass
            self.engine.run_event(Event.BEFORE_FORWARD)

            with get_precision_context(self.state.precision):
                self.state.outputs = self.state.model(self.state.batch)

            self.engine.run_event(Event.AFTER_FORWARD)

            # loss
            self.engine.run_event(Event.BEFORE_LOSS)

            with get_precision_context(self.state.precision):
                self.state.loss = self._original_model.loss(self.state.outputs, self.state.batch)

            assert self.state.loss is not None
            self.engine.run_event(Event.AFTER_LOSS)

            # backward
            self.engine.run_event(Event.BEFORE_BACKWARD)

            # Sum individual losses
            microbatch_loss = self._device.tensor_to_device(torch.zeros(size=(1,)))
            for loss in ensure_tuple(self.state.loss):
                microbatch_loss.add_(loss.mean())

            # Loss used for logging, scaled by grad_accum for correctly calculating metrics
            total_loss += microbatch_loss.detach().clone() * (microbatch_num_samples / current_batch_size)

            if use_grad_scaling:
                microbatch_loss = cast(torch.Tensor, self.state.scaler.scale(microbatch_loss))

            if self.deepspeed_enabled:
                self.state.deepspeed_model.backward(microbatch_loss)

            else:
                # Scale loss based on the number of samples in the microbatch to maintain gradient numerics
                microbatch_loss.mul_(microbatch_num_samples / current_batch_size)
                microbatch_loss.backward(create_graph=self._backwards_create_graph)

            self.engine.run_event(Event.AFTER_BACKWARD)

        if self.deepspeed_enabled:
            self.state.deepspeed_model.step()

    def predict(
        self,
        dataloader: Union[DataLoader, DataSpec],
        subset_num_batches: int = -1,
        *,
        return_outputs: bool = True,
    ):
        """Output model prediction on the provided data.

        There are two ways to access the prediction outputs.

        1.  With ``return_outputs`` set to True, the batch predictions will be collected into a list and returned.
        2.  Via a custom callback, which can be used with ``return_outputs`` set to False.

            This technique can be useful if collecting all the outputs from the dataloader would exceed available memory,
            and you want to write outputs directly to files. For example:

            .. testsetup::

                predict_dl = train_dataloader

            .. testcode::

                import os
                import torch

                from torch.utils.data import DataLoader

                from composer import Trainer, Callback
                from composer.loggers import Logger, LogLevel

                class PredictionSaver(Callback):
                    def __init__(self, folder: str):
                        self.folder = folder
                        os.makedirs(self.folder, exist_ok=True)

                    def predict_batch_end(self, state: State, logger: Logger) -> None:
                        name = f'batch_{int(state.predict_timestamp.batch)}.pt'
                        filepath = os.path.join(self.folder, name)
                        torch.save(state.outputs, filepath)

                        # Also log the outputs as an artifact
                        logger.file_artifact(LogLevel.BATCH, artifact_name=name, file_path=filepath)

                trainer = Trainer(
                    ...,
                    callbacks=PredictionSaver('./predict_outputs'),
                )

                trainer.predict(predict_dl, return_outputs=False)

                print(sorted(os.listdir('./predict_outputs')))

            .. testoutput::

                ['batch_1.pt', ...]

        Args:
            dataloader (DataLoader | DataSpec): The :class:`.DataLoader` or
                :class:`.DataSpec` for the prediction data.
            subset_num_batches (int, optional): If specified, only perform model prediction
                on this many batches. This parameter has no effect if it is greater than ``len(dataloader)``.
                If ``-1``, then the entire loader will be iterated over. (default: ``-1``)
            return_outputs (bool, optional): If True (the default), then prediction outputs will be (recursively)
                moved to cpu and accumulated into a list. Otherwise, prediction outputs are discarded after each
                batch.

        Returns:
            List: A list of batch outputs, if ``return_outputs`` is True. Otherwise, an empty list.

        """
        if isinstance(dataloader, DataSpec):
            data_spec = dataloader
        else:
            data_spec = DataSpec(dataloader)

        # Put the model into evaluation mode, but be able to restore it to training mode afterwards
        restore_model_train = self.state.model.training
        self.state.model.eval()

        # Bind the dataloader to the state, but be able to restore the previous dataloader afterwards
        original_dataloader = self.state.dataloader
        original_dataloader_label = self.state.dataloader_label
        original_dataloader_len = self.state.dataloader_len
        self.state.set_dataloader(data_spec.dataloader, 'predict', subset_num_batches)
        assert self.state.dataloader is not None, 'Already set the dataloader'

        # Reset the predict timestamp
        self.state.predict_timestamp = Timestamp()

        last_wct = datetime.datetime.now()

        outputs = []
        cpu_device = DeviceCPU()

        with torch.no_grad():

            self.engine.run_event(Event.PREDICT_START)

            for self.state.batch in self._iter_dataloader():
                # Move the batch onto the device
                self.state.batch = self._device.batch_to_device(self.state.batch)

                # Perform any device transforms
                if data_spec.device_transforms is not None:
                    self.state.batch = data_spec.device_transforms(self.state.batch)

                # Count the batch size and num tokens before any events run
                rank_num_samples = data_spec.get_num_samples_in_batch(self.state.batch)
                rank_num_tokens = data_spec.get_num_tokens_in_batch(self.state.batch)

                # Fix the batch if using DeepSpeed
                if self.deepspeed_enabled:
                    self.state.batch = _fix_batch_precision_for_deepspeed(self.state.batch, self.state.precision)

                self.engine.run_event(Event.PREDICT_BATCH_START)

                self.engine.run_event(Event.PREDICT_BEFORE_FORWARD)
                with get_precision_context(self.state.precision):
                    self.state.outputs = self.state.model(self.state.batch)
                self.engine.run_event(Event.PREDICT_AFTER_FORWARD)

                if return_outputs:
                    outputs.append(cpu_device.batch_to_device(self.state.outputs))

                now = datetime.datetime.now()
                batch_time = now - last_wct

                total_num_samples, total_num_tokens, batch_time = self._accumulate_time_across_ranks(
                    num_samples=rank_num_samples,
                    num_tokens=rank_num_tokens,
                    batch_time=batch_time,
                )

                last_wct = now

                self.state.predict_timestamp = self.state.predict_timestamp.to_next_batch(samples=total_num_samples,
                                                                                          tokens=total_num_tokens,
                                                                                          duration=batch_time)

                self.engine.run_event(Event.PREDICT_BATCH_END)

            self.engine.run_event(Event.PREDICT_END)

        # Restore training mode
        if restore_model_train:
            self.state.model.train()

        # Restore the dataloader
        self.state.set_dataloader(original_dataloader, original_dataloader_label)
        if original_dataloader_len is not None:
            self.state.dataloader_len = original_dataloader_len

        return outputs

    def eval(
        self,
        dataloader: Union[Iterable, DataSpec, dict],
        dataloader_label: str = 'eval',
        *,
        metrics: Union[Metric, MetricCollection],
        subset_num_batches: int = -1,
        log_level: Union[str, LogLevel] = LogLevel.FIT,
    ):
        """Evaluate the model and log appropriate metrics.

        Args:
            dataloader (DataLoader | DataSpec | dict): The class:`.DataLoader`, :class:`.DataSpec`, or
                dict of :class:`.DataSpec` kwargs to use for evaluation
            dataloader_label (str, optional): The dataloader label to use for logging metrics. Defaults to ``'eval'``.
            metrics (Metric | MetricCollection): The metrics to log.
            subset_num_batches (int, optional): If specified, evaluate on this many batches. Defaults to ``-1``,
                which means to iterate over the entire dataloader.

                This parameter has no effect if ``eval_dataloader`` is not specified, it is greater than
                ``len(eval_dataloader)``, or ``eval_dataloader`` is an :class:`.Evaluator` (which is via
                ``Evaluator(subset_num_batches=...)``.)
            log_level (LogLevel | str, optional): The log level to use when logging metrics. Defaults to
                :attr:`~.LogLevel.FIT`.
        """
        log_level = LogLevel(log_level)
        restore_model_train = self.state.model.training

        # back up the original dataloader on the state, so we can restore it after evaluation is finished
        original_dataloader = self.state.dataloader
        original_dataloader_label = self.state.dataloader_label
        original_num_batches = self.state.dataloader_len

        # Unpack the dataloader
        if isinstance(dataloader, dict):
            # treat as DataSpec kwargs
            dataloader = DataSpec(**dataloader)
        if not isinstance(dataloader, DataSpec):
            dataloader = DataSpec(dataloader)
        data_spec = dataloader

        # Reset the eval timestamp
        self.state.eval_timestamp = Timestamp()

        last_wct = datetime.datetime.now()

        self.state.model.eval()
        with torch.no_grad():
            self.state.set_dataloader(data_spec.dataloader, dataloader_label, subset_num_batches)
            assert self.state.dataloader is not None, 'dataloader is set'

            self.engine.run_event(Event.EVAL_START)

            if not isinstance(metrics, MetricCollection):
                metrics = MetricCollection(metrics)

            metrics = self._ensure_metrics_device_and_dtype(metrics)
            metrics.reset()
            dataloader = self.state.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                # The distributed sampler uses `set_epoch` to set the random seed
                # Because evaluation can run on each batch, we use the batch to seed the sampler
                # so each evaluation will get a proper shuffle.
                # The epoch provided to `set_epoch` need not be sequential, so this is fine.
                dataloader.sampler.set_epoch(int(self.state.timestamp.batch))

            for self.state.batch in self._iter_dataloader():
                self.state.batch = self._device.batch_to_device(self.state.batch)
                if data_spec.device_transforms is not None:
                    self.state.batch = data_spec.device_transforms(self.state.batch)

                # Count the batch size and num tokens before any events run
                rank_num_samples = data_spec.get_num_samples_in_batch(self.state.batch)
                rank_num_tokens = data_spec.get_num_tokens_in_batch(self.state.batch)

                if self.deepspeed_enabled:
                    self.state.batch = _fix_batch_precision_for_deepspeed(self.state.batch, self.state.precision)

                self.engine.run_event(Event.EVAL_BATCH_START)

                self.engine.run_event(Event.EVAL_BEFORE_FORWARD)
                with get_precision_context(self.state.precision):
                    self.state.outputs, targets = self._original_model.validate(self.state.batch)
                self.engine.run_event(Event.EVAL_AFTER_FORWARD)

                # Run in same precision context to avoid NaNs
                with get_precision_context(self.state.precision):
                    metrics.update(self.state.outputs, targets)

                now = datetime.datetime.now()
                batch_time = now - last_wct

                total_num_samples, total_num_tokens, batch_time = self._accumulate_time_across_ranks(
                    num_samples=rank_num_samples,
                    num_tokens=rank_num_tokens,
                    batch_time=batch_time,
                )

                self.state.eval_timestamp = self.state.eval_timestamp.to_next_batch(
                    samples=total_num_samples,
                    tokens=total_num_tokens,
                    duration=batch_time,
                )

                last_wct = now

                self.engine.run_event(Event.EVAL_BATCH_END)

            self.logger.data_epoch({'epoch': self.state.timestamp.epoch.value})
            self.logger.data_batch({'trainer/global_step': self.state.timestamp.batch.value})

            self._compute_and_log_metrics(dataloader_label=dataloader_label, metrics=metrics, log_level=log_level)

            self.engine.run_event(Event.EVAL_END)

        if restore_model_train:
            self.state.model.train()

        self.state.set_dataloader(original_dataloader, original_dataloader_label)
        if original_num_batches is not None:
            self.state.dataloader_len = original_num_batches

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

    def _iter_dataloader(self):
        """Helper method to iterate over the dataloader.

        This method yields up to :attr:`.State.dataloader_len`` batches from the dataloader. In addition, if the
        profiler is enabled, the dataloader latency recorded via the :class:`.Marker` API.
        """
        marker = None
        if self.state.profiler is not None:
            marker = self.state.profiler.marker(f'dataloader/{self.state.dataloader_label}', categories=['dataloader'])
        assert self.state.dataloader is not None, 'the dataloader should be set before calling this method'

        if self.state.dataloader_len is None:
            dataloader_iter = iter(self.state.dataloader)
        else:
            dataloader_iter = itertools.islice(self.state.dataloader, int(self.state.dataloader_len))

        while True:
            if marker is not None:
                marker.start()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            finally:
                if marker is not None:
                    marker.finish()
            yield batch

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
            raise RuntimeError('state.optimizers must be set before `_use_closures` can be determined')

        return all(
            getattr(optimizer, '_step_supports_amp_closure', False)
            for optimizer in ensure_tuple(self.state.optimizers))

    def save_checkpoint(self, name: str = 'ep{epoch}-ba{batch}-rank{rank}', *, weights_only: bool = False):
        """Checkpoint the training :class:`~.State`.

        Args:
            name (str, optional): See :func:`.save_checkpoint`.
            weights_only (bool, optional): See :func:`.save_checkpoint`.

        Returns:
            List[pathlib.Path]: See :func:`.save_checkpoint`.
        """
        return save_checkpoint(state=self.state, filename=name, weights_only=weights_only)

    def export_for_inference(
        self,
        save_format: Union[str, ExportFormat],
        save_path: str,
        save_object_store: Optional[ObjectStore] = None,
        sample_input: Optional[Any] = None,
        transforms: Optional[Sequence[Transform]] = None,
    ):
        """Export a model for inference.

        Args:
            save_format (Union[str, ExportFormat]):  Format to export to. Either ``"torchscript"`` or ``"onnx"``.
            save_path: (str): The path for storing the exported model. It can be a path to a file on the local disk,
            a URL, or if ``save_object_store`` is set, the object name
                in a cloud bucket. For example, ``my_run/exported_model``.
            save_object_store (ObjectStore, optional): If the ``save_path`` is in an object name in a cloud bucket
                (i.e. AWS S3 or Google Cloud Storage), an instance of
                :class:`~.ObjectStore` which will be used
                to store the exported model. If this is set to ``None``,  will save to ``save_path`` using the trainer's
                logger. (default: ``None``)
            sample_input (Any, optional): Example model inputs used for tracing. This is needed for "onnx" export.
                The ``sample_input`` need not match the batch size you intend to use for inference. However, the model
                should accept the ``sample_input`` as is. (default: ``None``)
            transforms (Sequence[Transform], optional): transformations (usually optimizations) that should
                be applied to the model. Each Transform should be a callable that takes a model and returns a modified model.

        Returns:
            None
        """
        export_model = self.state.model.module if self.state.is_model_ddp else self.state.model
        if not isinstance(export_model, nn.Module):
            raise ValueError(f'Exporting Model requires type torch.nn.Module, got {type(export_model)}')
        if sample_input == None and save_format == 'onnx':
            sample_input = self.state.batch
        export_with_logger(model=export_model,
                           save_format=save_format,
                           save_path=save_path,
                           logger=self.logger,
                           save_object_store=save_object_store,
                           sample_input=(sample_input,),
                           transforms=transforms)
