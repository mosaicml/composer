# Copyright 2021 MosaicML. All Rights Reserved.

"""Train models!"""

from __future__ import annotations

import contextlib
import datetime
import itertools
import logging
import pathlib
import warnings
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union, cast

import torch
import torch.distributed
import torch.utils.data
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Metric, MetricCollection

import composer
from composer.algorithms import ScaleSchedule
from composer.callbacks import CheckpointSaver
from composer.core import (Algorithm, Callback, DataSpec, Engine, Evaluator, Event, Precision, State, Time, Timestamp,
                           ensure_data_spec, ensure_evaluator, ensure_time)
from composer.core.precision import get_precision_context
from composer.core.types import Batch, BreakEpochException, PyTorchScheduler
from composer.loggers import Logger, LoggerDestination, LogLevel, ProgressBarLogger
from composer.models.base import ComposerModel
from composer.optim.decoupled_weight_decay import DecoupledSGDW
from composer.optim.scheduler import ComposerScheduler, compile_composer_scheduler
from composer.profiler import Profiler, ProfilerAction, SystemProfiler, TorchProfiler, TraceHandler
from composer.trainer._deepspeed import _fix_batch_precision_for_deepspeed, _parse_deepspeed_config
from composer.trainer._scale_schedule import scale_pytorch_scheduler
from composer.trainer._scaler import ClosureGradScaler
from composer.trainer.ddp import DDPSyncStrategy, ddp_sync_context, prepare_ddp_module
from composer.trainer.devices import Device, DeviceCPU, DeviceGPU
from composer.utils import dist, ensure_tuple, map_collection, module_surgery, reproducibility
from composer.utils.checkpoint import load_checkpoint, save_checkpoint
from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store import ObjectStore

log = logging.getLogger(__name__)

__all__ = ["Trainer"]

# syntax to shorten the Scheduler type annoations
Scheduler = Union[ComposerScheduler, PyTorchScheduler]


def _raise_missing_argument_exception(arg_name: str):
    raise ValueError((f"{arg_name} is a required argument and must be specified when constructing the "
                      f"{Trainer.__name__} or when calling {Trainer.__name__}.{Trainer.fit.__name__}(). "
                      f"To fix, please specify `arg_name` via {Trainer.__name__}({arg_name}=...) or "
                      f"{Trainer.__name__}.{Trainer.fit.__name__}({arg_name}=...)."))


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


def _should_step_schedulers_every_batch(
    schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]],
    step_schedulers_every_batch: Optional[bool],
):
    pytorch_schedulers = [
        scheduler for scheduler in ensure_tuple(schedulers) if isinstance(scheduler, PyTorchScheduler)
    ]
    if len(pytorch_schedulers) > 0:
        if step_schedulers_every_batch is True:
            log.info(("Schedulers are being steped every batch, as `step_schedulers_every_batch` is True. "
                      "The trainer cannot automatically convert the parameters (e.g. step_size, T_max) of the "
                      f"PyTorch {type(pytorch_schedulers[0]).__name__} scheduler to be in terms of batches. "
                      "Please ensure that the scheduler parameters are in terms of batches, not epochs. "
                      "Alternatively, use a ComposerScheduler. For more information, see "
                      f"https://docs.mosaicml.com/en/v{composer.__version__}/trainer/schedulers.html. "))

        else:
            if step_schedulers_every_batch is None:
                # only if all schedulers are ComposerSchedulers, then we can step every batch by default
                step_schedulers_every_batch = False

            log.info((
                "Schedulers will be stepped every epoch because the Trainer was constructed with a PyTorch "
                f"{type(pytorch_schedulers[0]).__name__} scheduler. To step the schedulers every batch, adjust the "
                "scheduler parameters (e.g. step_size, T_max) to be in terms of batches and set "
                "`step_schedulers_every_batch` to True, or alternatively use a ComposerScheduler. For more information, "
                f"see https://docs.mosaicml.com/en/v{composer.__version__}/trainer/schedulers.html."))

    else:
        step_schedulers_every_batch = True

    return step_schedulers_every_batch


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
        raise ValueError(f"{precision} is not supproted for CPU training.")
    if not deepspeed_enabled and precision == Precision.FP16:
        raise ValueError("FP16 precision is only supported when training with DeepSpeed.")


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


def _get_grad_accum(grad_accum: Union[int, str], device: Device) -> Tuple[int, bool]:
    # Set initial grad_accum to 1 if using adaptive
    adaptive_grad_accum = grad_accum == "auto"
    if adaptive_grad_accum:
        grad_accum = 1
        warnings.warn(("Setting `grad_accum='auto'` is an experimental feature which may cause "
                       "uncaught Cuda Out of Memory errors. In this case, please manually "
                       "set grad_accum explicitly to an integer instead. "))
    # Cannot use adaptive grad accum on CPU
    if isinstance(device, DeviceCPU) and adaptive_grad_accum:
        raise ValueError("Cannot use adaptive grad_accum on CPU. Please set grad_accum >= 1")
    # grad_accum should be int as we've already handled "auto" case
    if not isinstance(grad_accum, int):
        raise ValueError("grad_accum must be an int or ``'auto'``")
    return grad_accum, adaptive_grad_accum


def _is_cuda_oom(e: RuntimeError):
    """Determines if error is CUDA Out of Memory and if adaptive_grad_accum is enabled."""
    return "CUDA out of memory" in str(e)


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
        raise ValueError(f"Invalid seed: {seed}. It must be on [0; 2**32 - 1)")

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
            ddp_sync_strategy = DDPSyncStrategy.FORCED_SYNC
        else:
            ddp_sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC
    else:
        ddp_sync_strategy = DDPSyncStrategy(ddp_sync_strategy)
    return ddp_sync_strategy


def _initialize_profiler(
    state: State,
    prof_schedule: Optional[Callable[[State], ProfilerAction]],
    prof_trace_handlers: Optional[Union[TraceHandler, Sequence[TraceHandler]]],
    sys_prof_cpu: bool,
    sys_prof_memory: bool,
    sys_prof_disk: bool,
    sys_prof_net: bool,
    torch_prof_record_shapes: bool,
    torch_prof_profile_memory: bool,
    torch_prof_with_stack: bool,
    torch_prof_with_flops: bool,
    torch_prof_use_gzip: bool,
    torch_prof_overwrite: bool,
    torch_prof_num_traces_to_keep: int,
    torch_prof_artifact_name: str,
    torch_prof_folder: str,
    torch_prof_filename: str,
    sys_prof_stats_thread_interval_seconds: float,
):
    # Configure profilers if profiling is enabled

    profiler = None
    profiler_callbacks: List[Callback] = []
    if prof_schedule is not None or len(ensure_tuple(prof_trace_handlers)) > 0:
        if prof_schedule is None or len(ensure_tuple(prof_trace_handlers)) == 0:
            raise ValueError("To use the profiler, both `prof_schedule` and `prof_trans_handlers` must be specified.")

        profiler = Profiler(state=state, trace_handlers=ensure_tuple(prof_trace_handlers), schedule=prof_schedule)

        if sys_prof_cpu or sys_prof_memory or sys_prof_disk or sys_prof_net:
            profiler_callbacks.append(
                SystemProfiler(profile_cpu=sys_prof_cpu,
                               profile_memory=sys_prof_memory,
                               profile_disk=sys_prof_disk,
                               profile_net=sys_prof_net,
                               stats_thread_interval_seconds=sys_prof_stats_thread_interval_seconds))

        if torch_prof_record_shapes or torch_prof_profile_memory or torch_prof_with_stack or torch_prof_with_flops:
            profiler_callbacks.append(
                TorchProfiler(filename=torch_prof_filename,
                              folder=torch_prof_folder,
                              artifact_name=torch_prof_artifact_name,
                              num_traces_to_keep=torch_prof_num_traces_to_keep,
                              overwrite=torch_prof_overwrite,
                              record_shapes=torch_prof_record_shapes,
                              profile_memory=torch_prof_profile_memory,
                              use_gzip=torch_prof_use_gzip,
                              with_stack=torch_prof_with_stack,
                              with_flops=torch_prof_with_flops))

        # Append the trace handlers at the end, so profilers will log events before the traces are written.
        profiler_callbacks.extend(profiler.trace_handlers)

    state.profiler = profiler
    state.callbacks.extend(profiler_callbacks)


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
        train_dataloader (Iterable, DataSpec, or dict, optional): The dataloader, :class:`.DataSpec`,
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
        eval_interval (int | str | Time | (State, Event) -> bool, optional): An integer, which will be
            interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), a :class:`.Time` object, or a callable.
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
        run_name (str, optional): A name for this training run. If not specified, one will be generated automatically.

            .. seealso:: :class:`~composer.loggers.logger.Logger` for a description of the run name.
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
        load_object_store (ObjectStore, optional): If the ``load_path`` is in an object store
            (i.e. AWS S3 or Google Cloud Storage), an instance of :class:`.ObjectStore` which
            will be used to retreive the checkpoint. Otherwise, if the checkpoint is a local filepath,
            set to ``None``. Ignored if ``load_path`` is ``None``. (default: ``None``)

            Example:

            .. testsetup::

                import composer.trainer

                composer.trainer.trainer.load_checkpoint = lambda *args, **kwargs: None

            .. testcode::

                from composer import Trainer
                from composer.utils import ObjectStore

                # Create the object store provider with the specified credentials
                creds = {"key": "object_store_key",
                         "secret": "object_store_secret"}
                store = ObjectStore(provider="s3",
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

            .. testcleanup::

                trainer.engine.close()
        load_weights_only (bool, optional): Whether or not to only restore the weights from the checkpoint without
            restoring the associated state. Ignored if ``load_path`` is ``None``. (default: ``False``)
        load_strict (bool, optional): Ensure that the set of weights in the checkpoint and model must exactly match.
            Ignored if ``load_path`` is ``None``. (default: ``False``)
        load_chunk_size (int, optional): Chunk size (in bytes) to use when downloading checkpoints.
            Ignored if ``load_path`` is either ``None`` or a local file path. (default: ``1,048,675``)
        load_progress_bar (bool, optional): Display the progress bar for downloading the checkpoint.
            Ignored if ``load_path`` is either ``None`` or a local file path. (default: ``True``)
        save_folder (str, optional): Format string for the folder where checkpoints are saved.
            If ``None``, checkpoints will not be saved. (default: ``None``)

            .. seealso:: :class:`~.CheckpointSaver`

            .. note::

                For fine-grained control on checkpoint saving (e.g. to save different types of checkpoints
                at different intervals), leave this parameter as ``None``, and instead pass
                instance(s) of :class:`~.CheckpointSaver` directly as ``callbacks``.
        save_filename (str, optional): A format string describing how to name checkpoints.
            This parameter has no effect if ``save_folder`` is ``None``.
            (default: ``"ep{epoch}-ba{batch}-rank{rank}"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_artifact_name (str, optional): A format string describing how to name checkpoints in loggers.
            This parameter has no effect if ``save_folder`` is ``None``.
            (default: ``"{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}"``)

            .. seealso:: :class:`~.CheckpointSaver`
        save_latest_filename (str, optional): A format string for the name of a symlink
            (relative to ``save_folder``) that points to the last saved checkpoint.
            This parameter has no effect if ``save_folder`` is ``None``.
            To disable symlinking, set this to ``None``. (default: ``"latest-rank{rank}"``)

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
        deepspeed_config (Dict[str, Any], optional): Configuration for DeepSpeed, formatted as a JSON
            according to `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_. (default: ``None``)
            
            To use DeepSpeed with default values, set to the empty dictionary ``{}``.
            To disable DeepSpeed (the default), set to ``None``.
        device (Device | str, optional): The device to use for training, which can be ``'cpu'`` or ``'gpu'``.
            (default: ``None``)

            The default behavior sets the device to ``'gpu'`` if CUDA is available; otherwise, it sets the device to
            ``'cpu'``.
        precision (Precision | str, optional): Numerical precision to use for training. One of ``fp32``, ``fp16``
            or ``amp`` (recommended). (default: ``Precision.FP32``)

            .. note::
                ``fp16`` only works if ``deepspeed_config`` is also provided.
        grad_accum (Union[int, str], optional): The number of microbatches to split a per-device batch into. Gradients
            are summed over the microbatches per device. If set to ``auto``, dynamically increases grad_accum
            if microbatch is too large for GPU. (default: ``1``)

            .. note:: This is implemented by taking the batch yielded by the ``train_dataloader`` and splitting
                it into ``grad_accum`` sections. Each section is of size ``train_dataloader // grad_accum``.
                If the batch size of the dataloader is not divisible by ``grad_accum``,
                then the last section will be of size ``batch_size % grad_accum``.
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
        ddp_sync_strategy (str or DDPSyncStrategy, optional): The strategy to use for synchronizing gradients.
            Leave unset to let the trainer auto-configure this. See :class:`.DDPSyncStrategy`
            for more details.
        grad_clip_norm (float, optional): The norm to clip gradient magnitudes to. Set to ``-1`` for no gradient
            clipping. (default: ``-1``)
        prof_schedule ((State) -> ProfilerAction, optional): The profiler scheduler.

            Must be specified in conjunction with ``prof_trace_handlers`` to use the profiler.

            .. testcode::

                from composer.trainer import Trainer
                from composer.profiler import JSONTraceHandler, cyclic_schedule

                trainer = Trainer(
                    ...,
                    prof_trace_handlers=JSONTraceHandler(
                        folder='traces',
                    ),
                    prof_schedule=cyclic_schedule(
                        skip_first=1,
                        wait=0,
                        warmup=1,
                        active=4,
                        repeat=1,
                    ),
                )

            .. testcleanup::

                trainer.engine.close()

            .. seealso:: :mod:`composer.profiler` for more details on profiling with the trainer.

        prof_trace_handlers (TraceHandler | Sequence[TraceHandler], optional): Profiler trace handlers.

            Must be specified in conjunction with ``prof_trace_handlers`` to use the profiler.

            .. seealso:: :mod:`composer.profiler` for more details on profiling with the trainer.
        sys_prof_cpu (bool, optional): Whether to record cpu statistics.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified. (default: ``True``).
        sys_prof_memory (bool, optional): Whether to record memory statistics.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified. (default: ``False``).
        sys_prof_disk (bool, optional): Whether to record disk statistics.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified. (default: ``False``).
        sys_prof_net (bool, optional): Whether to record network statistics.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified. (default: ``False``).
        sys_prof_stats_thread_interval_seconds (float, optional): Interval to record stats, in seconds.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified. (default: ``0.5``).
        torch_prof_folder (str, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_filename (str, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_artifact_name (str, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_overwrite (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_use_gzip (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_record_shapes (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_profile_memory (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_with_stack (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_with_flops (bool, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.
        torch_prof_num_traces_to_keep (int, optional): See :class:`~composer.profiler.torch_profiler.TorchProfiler`.
            Ignored if ``prof_schedule`` and ``prof_trace_handlers`` are not specified.

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
        load_object_store: Optional[ObjectStore] = None,
        load_weights_only: bool = False,
        load_strict: bool = False,
        load_chunk_size: int = 1_048_576,
        load_progress_bar: bool = True,

        # Save Checkpoint
        save_folder: Optional[str] = None,
        save_filename: str = "ep{epoch}-ba{batch}-rank{rank}",
        save_artifact_name: str = "{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}",
        save_latest_filename: str = "latest-rank{rank}",
        save_latest_artifact_name: str = "{run_name}/checkpoints/latest-rank{rank}",
        save_overwrite: bool = False,
        save_interval: Union[str, int, Time, Callable[[State, Event], bool]] = "1ep",
        save_weights_only: bool = False,
        save_num_checkpoints_to_keep: int = -1,

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
        prof_schedule: Optional[Callable[[State], ProfilerAction]] = None,
        prof_trace_handlers: Optional[Union[TraceHandler, Sequence[TraceHandler]]] = None,
        sys_prof_cpu: bool = True,
        sys_prof_memory: bool = False,
        sys_prof_disk: bool = False,
        sys_prof_net: bool = False,
        sys_prof_stats_thread_interval_seconds: float = 0.5,
        torch_prof_folder: str = '{run_name}/torch_traces',
        torch_prof_filename: str = 'rank{rank}.{batch}.pt.trace.json',
        torch_prof_artifact_name: str = '{run_name}/torch_traces/rank{rank}.{batch}.pt.trace.json',
        torch_prof_overwrite: bool = False,
        torch_prof_use_gzip: bool = False,
        torch_prof_record_shapes: bool = False,
        torch_prof_profile_memory: bool = True,
        torch_prof_with_stack: bool = False,
        torch_prof_with_flops: bool = True,
        torch_prof_num_traces_to_keep: int = -1,
    ):
        # Determine whether DeepSpeed is enabled
        deepspeed_enabled = deepspeed_config is not None

        # Device
        self._device = _get_device(device)

        # Distributed
        if deepspeed_enabled or dist.get_world_size() > 1:
            # deepspeed requires torch.distributed to be initialized, even if the world size is 1
            # distributed is always required with multi-rank training
            dist.initialize_dist(self._device.dist_backend, datetime.timedelta(seconds=dist_timeout))

        # Reproducibility
        rank_zero_seed, seed = _distribute_and_get_random_seed(seed, self._device)
        # If hparams is used to create the Trainer this function is called twice
        # which is okay because all runs with the hparams codepath will do this
        log.info(f"Setting seed to {seed}")
        reproducibility.seed_all(seed)
        if deterministic_mode:
            reproducibility.configure_deterministic_mode()

        # Precision
        if precision is None:
            precision = Precision.AMP if isinstance(device, DeviceGPU) else Precision.FP32
        if isinstance(precision, str):
            precision = Precision(precision)

        _validate_precision(precision, self._device, deepspeed_enabled)

        # optimizers and schedulers
        if not optimizers:
            optimizers = DecoupledSGDW(list(model.parameters()), lr=0.1)
            warnings.warn(f"No optimizer was specified. Defaulting to {repr(optimizers)}")

        num_optimizers = len(ensure_tuple(optimizers))
        if num_optimizers != 1:
            raise NotImplementedError(f"Only one optimizer is supported; found {num_optimizers} optimizers")

        # Grad Accum
        grad_accum, self.adaptive_gradient_accumulation = _get_grad_accum(grad_accum, self._device)

        # Create the State
        self.state = State(
            rank_zero_seed=rank_zero_seed,
            algorithms=algorithms,
            model=model,
            callbacks=callbacks,
            grad_accum=grad_accum,
            precision=precision,
            optimizers=optimizers,
        )

        # Profiler
        _initialize_profiler(
            state=self.state,
            prof_schedule=prof_schedule,
            prof_trace_handlers=prof_trace_handlers,
            sys_prof_stats_thread_interval_seconds=sys_prof_stats_thread_interval_seconds,
            sys_prof_cpu=sys_prof_cpu,
            sys_prof_disk=sys_prof_disk,
            sys_prof_memory=sys_prof_memory,
            sys_prof_net=sys_prof_net,
            torch_prof_artifact_name=torch_prof_artifact_name,
            torch_prof_filename=torch_prof_filename,
            torch_prof_folder=torch_prof_folder,
            torch_prof_overwrite=torch_prof_overwrite,
            torch_prof_record_shapes=torch_prof_record_shapes,
            torch_prof_profile_memory=torch_prof_profile_memory,
            torch_prof_with_stack=torch_prof_with_stack,
            torch_prof_with_flops=torch_prof_with_flops,
            torch_prof_use_gzip=torch_prof_use_gzip,
            torch_prof_num_traces_to_keep=torch_prof_num_traces_to_keep,
        )

        # Console Logging
        loggers = list(ensure_tuple(loggers))
        if any(isinstance(x, ProgressBarLogger) for x in loggers):
            warnings.warn(
                DeprecationWarning(
                    (f"Specifying the {ProgressBarLogger.__name__} via `loggers` is deprecated. Instead, "
                     "please specify `progress_bar`, `log_to_console`, `log_level`, and `stream` arguments when "
                     "constructing the trainer. If specified, these arguments will be ignored, as the "
                     f"{ProgressBarLogger.__name__} was already created.")))
        else:
            loggers.append(
                ProgressBarLogger(
                    progress_bar=progress_bar,
                    log_to_console=log_to_console,
                    console_log_level=console_log_level,
                    stream=console_stream,
                ))

        # Logger
        self.logger = Logger(state=self.state, destinations=loggers, run_name=run_name)

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
        self.train_metrics = _get_training_metrics(model) if compute_training_metrics else None

        # Max Duration
        if max_duration is not None:
            self.state.max_duration = ensure_time(max_duration)

        self.logger.data_fit({"rank_zero_seed": rank_zero_seed})

        assert isinstance(self.state.model, ComposerModel)
        self._original_model = self.state.model

        # Schedulers
        # ScaleSchedule is a deprecated algorithm, but if it is used, updated SSR with its ratio.
        # TODO(#434): Remove this completely.
        for algorithm in ensure_tuple(algorithms):
            if isinstance(algorithm, ScaleSchedule):
                scale_schedule_ratio = algorithm.ratio
        self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)
        if scale_schedule_ratio != 1.0:
            if len(self.state.schedulers) == 0:
                raise ValueError("Specifying `scale_schedule_ratio` without `schedulers` has no effect.")
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)

        self._step_schedulers_every_batch = _should_step_schedulers_every_batch(schedulers, step_schedulers_every_batch)

        if len(ensure_tuple(schedulers)) == 0:
            warnings.warn(f"NoSchedulerWarning: No schedulers were specified. The learning rate will be constant.")

        # Evaluators
        if eval_dataloader is None:
            self.evaluators: List[Evaluator] = []
        else:
            self.evaluators = [
                ensure_evaluator(evaluator, model.metrics(train=False)) for evaluator in ensure_tuple(eval_dataloader)
            ]
            _set_evaluator_interval_and_subset_num_batches(
                evaluators=self.evaluators,
                eval_interval=eval_interval,
                subset_num_batches=eval_subset_num_batches,
            )
        if len(self.evaluators) == 0:
            if eval_subset_num_batches != -1:
                raise ValueError("Specifying `eval_subset_num_batches` without an `eval_dataloader` has no effect.")
            if eval_interval != 1:
                raise ValueError("Specifying `eval_interval` without an `eval_dataloader` has no effect.")

        # Grad Clip Norm
        self._grad_clip_norm = grad_clip_norm

        # Some algorithms require specific settings
        self._backwards_create_graph = any(map(lambda x: x.backwards_create_graph, ensure_tuple(algorithms)))
        self._find_unused_parameters = any(map(lambda x: x.find_unused_parameters, ensure_tuple(algorithms)))
        self._ddp_sync_strategy = _get_ddp_sync_strategy(ddp_sync_strategy, self._find_unused_parameters)

        # Configure Deepspeed
        if deepspeed_config is not None:
            try:
                import deepspeed
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group="deepspeed",
                                                    conda_package="deepspeed>=0.5.5") from e
            deepspeed_config = _parse_deepspeed_config(
                deepspeed_config,
                state=self.state,
                grad_clip_norm=self._grad_clip_norm,
            )
            optimizer = ensure_tuple(self.state.optimizers)[0]
            (self.state.model, self.state.optimizers, _, _) = deepspeed.initialize(
                config=deepspeed_config,
                model=self.state.model,
                optimizer=optimizer,
            )
            # Since the DeepSpeed ZeRO optimizer does not inherit torch.optim.Optimizer, the schedulers must be
            # compiled and bound BEFORE DeepSpeed initialization. However, this is OK, as the the DeepSpeed Zero
            # optimizer uses the same underlying parameter groups as the original optimizer. See
            # * https://github.com/microsoft/DeepSpeed/blob/fee73135980e78f8be7e1a3ff556751623ef6aaa/deepspeed/runtime/zero/stage_1_and_2.py#L1911-L1917
            # * https://github.com/microsoft/DeepSpeed/blob/ef17c89570ceae5b26a5f886e9d8cd0941afc0ac/deepspeed/runtime/zero/stage3.py#L2532-L2538
            # In addition, the deepspeed engine is responsible for serializing the model and optimizer state,
            # so these attributes should not be serialized with the composer state.
            if "model" in self.state.serialized_attributes:
                self.state.serialized_attributes.remove("model")

            if "optimizers" in self.state.serialized_attributes:
                self.state.serialized_attributes.remove("optimizers")

        # If using DeepSpeed, the model must be loaded from checkpoint after the engine has been
        # initialized, but if using PyTorch DDP, the model must be loaded before it is wrapped with
        # DDP.

        # Load Checkpoint
        self._rng_state = None
        if load_path is not None:
            self._rng_state = load_checkpoint(state=self.state,
                                              path=load_path,
                                              object_store=load_object_store,
                                              load_weights_only=load_weights_only,
                                              strict_model_weights=load_strict,
                                              chunk_size=load_chunk_size,
                                              progress_bar=load_progress_bar)
            log.info(f"Setting seed to {self.state.seed}")
            reproducibility.seed_all(self.state.seed)

        # Move the model and optimizers to the specified device
        if not self.state.is_model_deepspeed:
            host_model_params = self.state.model.parameters()
            self.state.model = self._device.module_to_device(self.state.model)
            device_model_params = self.state.model.parameters()

            # use surgery to update the parameters of the optimizers, now that the model is on the device
            # see https://pytorch.org/docs/stable/optim.html#constructing-it
            module_surgery.replace_params_in_optimizer(old_params=host_model_params,
                                                       new_params=device_model_params,
                                                       optimizers=self.state.optimizers)

            # Move any remaining optimizer parameters onto the device
            self.state.optimizers = map_collection(self.state.optimizers, self._device.optimizer_to_device)

            if dist.get_world_size() > 1:
                # Only wrap the module if required
                self.state.model = prepare_ddp_module(self.state.model, self._find_unused_parameters)

    @property
    def deepspeed_enabled(self):
        """Whether DeepSpeed is enabled.

        .. seealso:: `DeepSpeed's documentation <https://www.deepspeed.ai/docs/config-json/>`_
        """
        return self.state.is_model_deepspeed

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

    def fit(
        self,
        *,
        # Train Dataloader
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = "train",
        train_subset_num_batches: Optional[int] = None,
        compute_training_metrics: Optional[bool] = None,

        # Timing
        duration: Optional[Union[int, str, Time[int]]] = None,
        reset_timer: bool = False,

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

        # Grad Clipping
        grad_clip_norm: Optional[float] = None,
    ):
        """Train the model.

        .. note::

            All arguments to :meth:`.fit` are optional. Any values specified here will override
            what was provided when constructing the :class:`.Trainer`.

        Args:
            train_dataloader (Iterable | DataSpec | Dict[str, Any], optional): See :class:`.Trainer`.
            train_dataloader_label (str, optional): See :class:`.Trainer`.
            train_subset_num_batches (int, optional): See :class:`.Trainer`.
            compute_training_metrics (bool, optional): See :class:`.Trainer`.
            reset_timer (bool): Whether to reset the :attr:`.State.timer`. Defaults to False.

                If ``True``, the timer will be zeroed out, causing :class:`.ComposerScheduler` and :class:`.Algorithm`
                instances to start from the beginning, as if it is a new training run.
                The :attr:`~.State.max_duration` will be incremented by the ``duration`` parameter.

                .. note::

                    Model gradients, optimizer states, and native PyTorch schedulers will not be reset.

                If ``False`` (the default), the timer will resume from where the previous call to :meth:`.fit`
                finished (or from zero, if a new training run).
                The :attr:`~.State.max_duration` will set to the ``duration`` parameter.

            duration (Time[int] | str | int, optional): The duration to train. Can be an integer, which will be
                interpreted to be epochs, a str (e.g. ``1ep``, or ``10ba``), or a :class:`.Time` object.

                If ``reset_timer`` is False (the default), then :attr:`.State.max_duration` will be converted
                into the same units as this parameter (if necessary), and then the max duration incremented by the
                value of this parameter.

                If ``reset_timer`` is True, then :attr:`.State.max_duration` will be set to this parameter.

            optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional): See :class:`.Trainer`.
            schedulers (PyTorchScheduler | ComposerScheduler | Sequence[PyTorchScheduler | ComposerScheduler], optional):
                See :class:`.Trainer`.
            scale_schedule_ratio (float, optional): See :class:`.Trainer`.
            step_schedulers_every_batch (bool, optional): See :class:`.Trainer`.
            eval_dataloader (Iterable | DataSpec | Evaluator | Sequence[Evaluator], optional): See :class:`.Trainer`.
            eval_subset_num_batches (int, optional): See :class:`.Trainer`.
            eval_interval (int | str | Time | (State, Event) -> bool, optional): See :class:`.Trainer`.
            grad_accum (int | str, optional): See :class:`.Trainer`.
            precision (Precision | str, optional): See :class:`.Trainer`.
            grad_clip_norm (float, optional): See :class:`.Trainer`.
        """
        # Train Dataloader
        if train_dataloader is not None:
            self._train_data_spec = ensure_data_spec(train_dataloader)
            self.state.set_dataloader(self._train_data_spec.dataloader, train_dataloader_label)
        if self._train_data_spec is None:
            _raise_missing_argument_exception("train_dataloader")
        if train_subset_num_batches is not None:
            self.state.dataloader_len = train_subset_num_batches
        if compute_training_metrics is not None:
            self.train_metrics = _get_training_metrics(self._original_model) if compute_training_metrics else None

        # Reset Timer
        if reset_timer:
            self.state.timer.reset()

        # Max Duration
        if duration is not None:
            duration = ensure_time(duration)
            # Effectively increment the max duration (if not resetting the Timer)
            # or set the max_duration (if resetting the timer -- self.state.timer.get(duration.unit) will be 0)
            # It is important to set the duration, rather than incrementing it, as ``duration`` could be in
            # different units than ``max_duration``
            self.state.max_duration = duration + self.state.timer.get(duration.unit)

        if self.state.max_duration is None:
            _raise_missing_argument_exception("max_duration")

        if self.state.max_duration <= self.state.timer.get(self.state.max_duration.unit) and not reset_timer:
            raise ValueError(
                (f"The max_duration ({self.state.max_duration}) is less than the elapsed training duration "
                 f"({self.state.timer.get(self.state.max_duration.unit)}). No training would occur. "
                 "Please either increase the `max_duration` or specify `reset_timer=True` in "
                 f"Trainer.fit()."))

        # Scale Schedule Ratio and Schedulers
        if scale_schedule_ratio != 1.0:
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)
        if schedulers is not None:
            self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)
            self._step_schedulers_every_batch = _should_step_schedulers_every_batch(schedulers,
                                                                                    step_schedulers_every_batch)
        else:
            if scale_schedule_ratio != 1.0:
                raise ValueError("Specifying `scale_schedule_ratio` without `schedulers` has no effect.")

            if step_schedulers_every_batch is not None:
                raise ValueError("Specifying `step_schedulers_every_batch` without `schedulers` has no effect.")

            if step_schedulers_every_batch is not None:
                raise ValueError("Specifying `step_schedulers_every_batch` without `schedulers` has no effect.")

        if len(ensure_tuple(schedulers)) == 0:
            warnings.warn(f"NoSchedulerWarning: No schedulers were specified. The learning rate will be constant.")

        # Evaluators
        if eval_dataloader is not None:
            self.evaluators = [
                # Need to use the `original_model` rather than `state.model`, as `state.model`
                # could be DDP / DeepSpeed wrapped.
                ensure_evaluator(evaluator, default_metrics=self._original_model.metrics(train=False))
                for evaluator in ensure_tuple(eval_dataloader)
            ]
            _set_evaluator_interval_and_subset_num_batches(
                evaluators=self.evaluators,
                eval_interval=eval_interval,
                subset_num_batches=eval_subset_num_batches,
            )
            if len(self.evaluators) == 0:
                if eval_subset_num_batches != -1:
                    raise ValueError("Specifying `eval_subset_num_batches` without an `eval_dataloader` has no effect.")
                if eval_interval != 1:
                    raise ValueError("Specifying `eval_interval` without an `eval_dataloader` has no effect.")

        if len(self.evaluators) == 0:
            warnings.warn(("No `eval_dataloader` was specified. Please specify `eval_dataloader` to periodically "
                           "evaluate your model while training."))

        # Grad Accum
        if grad_accum is not None:
            self.state.grad_accum, self.adaptive_gradient_accumulation = _get_grad_accum(grad_accum, self._device)

        # Grad Clip Norm
        if grad_clip_norm is not None:
            if self.state.is_model_deepspeed:
                raise ValueError("Changing the grad_clip_norm when using DeepSpeed is not supported.")
            self._grad_clip_norm = grad_clip_norm

        # Precision
        if precision is not None:
            if self.state.is_model_deepspeed:
                raise ValueError("Changing the precision when using DeepSpeed is not supported")
            precision = Precision(precision)
            _validate_precision(precision, self._device, self.state.is_model_deepspeed)
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
        for evaluator in self.evaluators:
            dataloader = evaluator.dataloader.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(0)
            for _ in dataloader:
                break

        # spin the train dataloader's sampler to get to the state of the desired epoch
        dataloader = self.state.dataloader
        assert dataloader is not None, "train dataloader is set on state after FIT_START"
        for epoch in range(int(self.state.timer.epoch)):
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
            for _ in dataloader:
                break

    def _train_loop(self) -> None:
        """Run training for the specified number of epochs and log results."""
        # print training start
        self.logger.data_fit({"trainer/algorithms": [str(algo) for algo in self.state.algorithms]})

        assert self.state.dataloader is not None, "dataloader is set in __init__() or fit()"
        assert self._train_data_spec is not None, "The train data spec is set in __init__() or fit()"

        self.engine.run_event(Event.FIT_START)

        # surpressing GradScaler warnings as they are always created
        # self._use_grad_scaling() will raise a RuntimeError if grad scaling is not available when it is required
        warnings.filterwarnings(action="ignore", message="torch.cuda.amp.GradScaler")
        self.state.scaler = ClosureGradScaler() if self._use_closures() else GradScaler()
        use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

        self._spin_dataloaders()

        if self.state.timer.batch_in_epoch == 0 and self._rng_state is not None:
            # only restore the rng state here if the step in the current epoch is zero.
            reproducibility.load_rng_state(self._rng_state)
            self._rng_state = None

        if self.train_metrics is not None:
            self.train_metrics = self._ensure_metrics_device_and_dtype(self.train_metrics)

        while self.state.timer < self.state.max_duration:
            try:
                self.state.model.train()

                if int(self.state.timer.batch_in_epoch) == 0:
                    self.engine.run_event(Event.EPOCH_START)
                    self.logger.data_epoch({"epoch": int(self.state.timer.epoch)})
                    if self.train_metrics is not None:
                        # reset the metrics before every epoch
                        self.train_metrics.reset()

                dataloader = self.state.dataloader
                if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                    dataloader.sampler.set_epoch(int(self.state.timer.epoch))

                for batch_idx, self.state.batch in enumerate(self._iter_dataloader()):

                    # if resuming, skip dataloader forward to the minibatch index
                    if batch_idx < int(self.state.timer.batch_in_epoch):
                        # Restore the RNG state immediately before the next batch is yielded from the dataloader
                        if batch_idx + 1 == int(self.state.timer.batch_in_epoch) and self._rng_state is not None:
                            reproducibility.load_rng_state(self._rng_state)
                            self._rng_state = None
                        continue

                    self.state.batch = self._device.batch_to_device(self.state.batch)
                    self.state.batch = self._train_data_spec.device_transforms(self.state.batch)
                    self.state.batch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
                    self.state.batch_num_tokens = self._train_data_spec.get_num_tokens_in_batch(self.state.batch)

                    if self.state.is_model_deepspeed:
                        self.state.batch = _fix_batch_precision_for_deepspeed(self.state.batch, self.state.precision)

                    if self.train_metrics is not None:
                        self.state.model.eval()
                        with torch.no_grad():
                            for eval_microbatch in self._train_data_spec.split_batch(
                                    self.state.batch, self.state.grad_accum):
                                # TODO: Detect if self.run_event(Event.AFTER_DATALOADER) changes the training
                                # data and if so print a warning that metrics may return unexpected results
                                self.train_metrics.update(*self._original_model.validate(eval_microbatch))

                    self.state.model.train()

                    self.engine.run_event(Event.AFTER_DATALOADER)

                    num_samples_in_batch = self._device.tensor_to_device(
                        torch.tensor([self.state.batch_num_samples], dtype=torch.int))
                    num_tokens_in_batch = self._device.tensor_to_device(
                        torch.tensor([self.state.batch_num_tokens], dtype=torch.int))
                    dist.all_reduce(num_samples_in_batch, reduce_operation="SUM")
                    dist.all_reduce(num_tokens_in_batch, reduce_operation="SUM")

                    self.engine.run_event(Event.BATCH_START)
                    self.logger.data_batch({
                        "trainer/global_step": int(self.state.timer.batch),
                        "trainer/batch_idx": self.state.timer.batch_in_epoch.value,
                    })

                    total_loss = self._train_batch(use_grad_scaling)

                    if use_grad_scaling:
                        self.state.scaler.update()

                    if total_loss is not None:
                        if not isinstance(total_loss, torch.Tensor):
                            total_loss = self._device.tensor_to_device(torch.tensor([total_loss]))

                        # total_loss can be None if gradient scaling failed
                        dist.all_reduce(total_loss, reduce_operation="SUM")
                        full_loss = total_loss.cpu().item()
                        self.logger.data_batch({'loss/train': full_loss / dist.get_world_size()})

                    self.state.timer.on_batch_complete(
                        samples=int(num_samples_in_batch.item()),
                        tokens=int(num_tokens_in_batch.item()),
                    )

                    if self._step_schedulers_every_batch:
                        for scheduler in self.state.schedulers:
                            scheduler.step()

                    if self.train_metrics is not None:
                        self._compute_and_log_metrics(
                            dataloader_label='train',
                            log_level=LogLevel.BATCH,
                            metrics=self.train_metrics,
                        )

                    self.engine.run_event(Event.BATCH_END)

                    for evaluator in self.evaluators:
                        assert evaluator.eval_interval is not None, "eval_interval should have been set on __init__() or fit()"
                        assert evaluator.subset_num_batches is not None, "subset_num_batches should have been set on __init__() or fit()"
                        if evaluator.eval_interval(self.state, Event.BATCH_END):
                            self.eval(
                                dataloader=evaluator.dataloader,
                                dataloader_label=evaluator.label,
                                subset_num_batches=evaluator.subset_num_batches,
                                metrics=evaluator.metrics,
                                log_level=LogLevel.BATCH,
                            )

                    self.engine.run_event(Event.BATCH_CHECKPOINT)

                    if self.state.timer >= self.state.max_duration:
                        # If max_duration is specified in batches, samples, or tokens, and
                        # and the max_duration is reached mid-epoch, then break out of the dataloader
                        # to finish the epoch early and finish training.
                        break

            except BreakEpochException:
                log.info(f'Skipping the rest of Epoch {int(self.state.timer.epoch)}')

            self.state.timer.on_epoch_complete()

            if self.train_metrics is not None:
                self._compute_and_log_metrics(
                    dataloader_label='train',
                    log_level=LogLevel.EPOCH,
                    metrics=self.train_metrics,
                )

            if not self._step_schedulers_every_batch:
                for scheduler in self.state.schedulers:
                    scheduler.step()

            self.engine.run_event(Event.EPOCH_END)

            for evaluator in self.evaluators:
                assert evaluator.eval_interval is not None, "eval_interval should have been set on __init__() or fit()"
                assert evaluator.subset_num_batches is not None, "subset_num_batches should have been set on __init__() or fit()"
                if evaluator.eval_interval(self.state, Event.EPOCH_END):
                    self.eval(
                        dataloader=evaluator.dataloader,
                        dataloader_label=evaluator.label,
                        subset_num_batches=evaluator.subset_num_batches,
                        metrics=evaluator.metrics,
                        log_level=LogLevel.EPOCH,
                    )

            self.engine.run_event(Event.EPOCH_CHECKPOINT)

    def _handle_cuda_oom(self):
        """Handles CUDA Out of Memory and rescales if using adaptive grad_accum."""
        # Raise runtime error if training 1 sample at a time still resulted in CUDA out of memory
        if self.state.grad_accum == self.state.batch_num_samples:
            raise RuntimeError(("CUDA out of memory. The train loop failed with an internal microbatch of size 1."
                                "The GPU does not have enough memory to process even 1 sample."))
        else:
            self.state.grad_accum = min(2 * self.state.grad_accum, self.state.batch_num_samples)
            self.logger.data_batch({'trainer/grad_accum': self.state.grad_accum})

    def _train_batch(self, use_grad_scaling: bool):
        """Compute loss by training on a full batch of data. Adaptively change microbatch size if enabled to maximize
        GPU usage.

        Args:
            use_grad_scaling (bool): Enables gradient scaling
        """
        assert self._train_data_spec is not None, "The train data spec should be set on __init__ or fit()"

        # Retry until we successfully complete training and return loss
        while True:
            total_loss = None
            # Note: We use uint8 instead of bool as BOR is not supported on all torch.distributed backends
            should_handle_cuda_oom = 0
            caught_timeout_error = None
            try:
                assert self.state.scaler is not None
                microbatches = self._train_data_spec.split_batch(self.state.batch, self.state.grad_accum)
                if self.state.is_model_deepspeed:
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
                if _is_cuda_oom(e):
                    log.debug((f"Rank {dist.get_global_rank()} OOM'd. "
                               "grad_accum will be increased prior to reattempting training on the current batch."))
                    should_handle_cuda_oom = 1
                elif "Timed out" in str(e):
                    # Catch timeout errors and only reraise if we did not encounter OOM on other ranks. Error
                    # is likely transient if one rank OOMed, it likely did not reach a barrier. Note that if we
                    # catch non-transient timeout errors they will be later reraised if no rank OOMed.
                    caught_timeout_error = e
                else:
                    raise

            # Propagate across all ranks if any rank hit CUDA OOM
            should_handle_cuda_oom = self._device.tensor_to_device(
                torch.tensor([should_handle_cuda_oom], dtype=torch.uint8))
            dist.all_reduce(should_handle_cuda_oom, reduce_operation="MAX")
            if int(should_handle_cuda_oom.item()) == 1:
                # If any rank hit CUDA OOM, update grad_accum and retry. Ignore any caught_timeout_error since
                # it is likely transient, e.g. timeout because certain ranks OOMed and didn't reach barrier.
                self._handle_cuda_oom()
            elif caught_timeout_error:
                # If not CUDA out of memory, raise exception to user. Note that this truncates the call stack
                # back only to this newly raised error.
                raise caught_timeout_error
            else:
                # Otherwise, return calculated loss
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

            if not self.state.is_model_deepspeed:
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

            # clip gradients if the magnitude is too large
            if not self.state.is_model_deepspeed and self._grad_clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.state.model.parameters(),
                    max_norm=self._grad_clip_norm,
                )

            self.engine.run_event(Event.AFTER_TRAIN_BATCH)

            return total_loss

    def _train_microbatch(self, use_grad_scaling: bool, current_batch_size: int, total_loss: torch.Tensor,
                          is_final_microbatch: bool):
        """Train and compute the loss of ``state.batch``, which is assumed to be a single microbatch.

        Args:
            use_grad_scaling (bool): Whether to use gradient scaling.
            minibatch_num_samples (int): Number of samples in the minibatch.
            total_loss (torch.Tensor): Total loss aggregated across all microbatches.
            is_final_microbatch (bool): If current microbatch is the last one.
        """
        assert self.state.scaler is not None
        assert self._train_data_spec is not None

        microbatch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
        sync_context = contextlib.nullcontext() if self.state.is_model_deepspeed else ddp_sync_context(
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

            # We always want to scale loss by the grad_accum before the backwards pass and
            # also for sake of metrics. Complicating matters, the DeepSpeed engine does its
            # own scaling when we call `.backward`, but this isn't in place so we still need
            # to scale for sake of metrics after the `.backward` call.

            # Loss is added to losses with clone to not scale the loss for the step printout
            # Likely need to look into the performance impact
            if not self.state.is_model_deepspeed:
                for loss in ensure_tuple(self.state.loss):
                    loss.mul_(microbatch_num_samples / current_batch_size)
                    total_loss += loss.detach().clone()

            assert self.state.loss is not None
            self.engine.run_event(Event.AFTER_LOSS)

            # backward
            self.engine.run_event(Event.BEFORE_BACKWARD)

            if use_grad_scaling:
                self.state.loss = cast(torch.Tensor, self.state.scaler.scale(self.state.loss))

            if self.state.is_model_deepspeed:
                self.state.deepspeed_model.backward(self.state.loss)

                # This is the same loss scaling and reporting we skipped earlier.
                for loss in ensure_tuple(self.state.loss):
                    loss.mul_(microbatch_num_samples / current_batch_size)
                    total_loss += loss.detach().clone()
            else:
                for loss in ensure_tuple(self.state.loss):
                    loss.backward(create_graph=self._backwards_create_graph)

            self.engine.run_event(Event.AFTER_BACKWARD)

        if self.state.is_model_deepspeed:
            self.state.deepspeed_model.step()

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

        self.state.model.eval()
        with torch.no_grad():
            self.state.set_dataloader(data_spec.dataloader, dataloader_label, subset_num_batches)
            assert self.state.dataloader is not None, "dataloader is set"

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
                dataloader.sampler.set_epoch(int(self.state.timer.batch))

            for self.state.batch in self._iter_dataloader():
                self.state.batch = self._device.batch_to_device(self.state.batch)
                if data_spec.device_transforms is not None:
                    self.state.batch = data_spec.device_transforms(self.state.batch)
                self.state.batch_num_samples = data_spec.get_num_samples_in_batch(self.state.batch)
                self.state.batch_num_tokens = data_spec.get_num_tokens_in_batch(self.state.batch)

                if self.state.is_model_deepspeed:
                    self.state.batch = _fix_batch_precision_for_deepspeed(self.state.batch, self.state.precision)

                self.engine.run_event(Event.EVAL_BATCH_START)

                self.engine.run_event(Event.EVAL_BEFORE_FORWARD)
                self.state.outputs, targets = self._original_model.validate(self.state.batch)
                self.engine.run_event(Event.EVAL_AFTER_FORWARD)

                metrics.update(self.state.outputs, targets)
                self._compute_and_log_metrics(dataloader_label=dataloader_label, metrics=metrics, log_level=log_level)

                self.engine.run_event(Event.EVAL_BATCH_END)

            self.logger.data_epoch({"epoch": self.state.timer.epoch.value})
            self.logger.data_batch({"trainer/global_step": self.state.timer.batch.value})

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
        if self.state.is_model_deepspeed:
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
            marker = self.state.profiler.marker(f"dataloader/{self.state.dataloader_label}", categories=["dataloader"])
        assert self.state.dataloader is not None, "the dataloader should be set before calling this method"

        if self.state.dataloader_len is None:
            dataloader_iter = iter(self.state.dataloader)
        else:
            dataloader_iter = itertools.islice(self.state.dataloader, int(self.state.dataloader_len))

        while True:
            if marker is not None:
                marker.start()
            try:
                yield next(dataloader_iter)
            except StopIteration:
                break
            finally:
                if marker is not None:
                    marker.finish()

    def _use_closures(self) -> bool:
        """Determines based on precision and optimizers whether to use closures.

        We default to using closures unless AMP is enabled, in which case we only allow closures when using optimizers
        with the _step_supports_amp_closure flag.
        """
        if self.state.is_model_deepspeed:
            return False

        if self.state.precision != Precision.AMP:
            return True

        if self.state.optimizers is None:
            raise RuntimeError("state.optimizers must be set before `_use_closures` can be determined")

        return all(
            getattr(optimizer, "_step_supports_amp_closure", False)
            for optimizer in ensure_tuple(self.state.optimizers))

    def save_checkpoint(self, name: str = "ep{epoch}-ba{batch}-rank{rank}", *, weights_only: bool = False):
        """Checkpoint the training :class:`~.State`.

        Args:
            name (str, optional): See :func:`.save_checkpoint`.
            weights_only (bool, optional): See :func:`.save_checkpoint`.

        Returns:
            List[pathlib.Path]: See :func:`.save_checkpoint`.
        """
        return save_checkpoint(state=self.state, logger=self.logger, filename=name, weights_only=weights_only)
