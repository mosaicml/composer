# Copyright 2021 MosaicML. All Rights Reserved.

"""Profiler to collect :mod:`torch` performance metrics during training."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

import torch.profiler
from torch.profiler.profiler import ProfilerAction as TorchProfilerAction

from composer.core import Callback, State
from composer.loggers import Logger
from composer.loggers.logger import LogLevel
from composer.profiler._profiler_action import ProfilerAction
from composer.utils import dist, ensure_folder_is_empty, format_name_with_dist, format_name_with_dist_and_time

__all__ = ["TorchProfiler"]

_PROFILE_MISSING_ERROR = "The profiler has not been setup. Please call profiler.init() before training starts."


class TorchProfiler(Callback):
    """Profile the execution using :class:`torch.profiler.profile`, implemented as a Composer
    :class:`~composer.core.callback.Callback`.    

    Profiling results are stored in TensorBoard format in the ``tensorboard_trace_handler_dir`` folder.

    When used with the Composer :class:`.Trainer`\\, profiling is enabled only if the ``tensorboard_trace_handler_dir`` is provided.

    .. note:: 
        
        The Composer :class:`.Trainer` creates an instance of :class:`.TorchProfiler` when ``tensorboard_trace_handler_dir`` is provided.
        The user should not create and directly register an instance of :class:`.TorchProfiler` when using the Composer :class:`.Trainer`\\.

    To view profiling results, run::

        pip install tensorbaord torch_tb_profiler
        tensorboard --logdir tensorboard_trace_handler_dir

    .. note::

        See :doc:`profiler` for additional usage details on :class:`torch.profiler.profile`\\.

    .. note::

        Enabling shape and stack tracing results in additional overhead.
        When ``record_shapes=True`` is specified, the profiler will temporarily hold references to tensors which
        may prevent certain optimizations that depend on the reference count and can introduce extra tensor copies.

    Args:
        filename_format (str): Format string for trace files.
        artifact_name_format (str, optional): Format string for trace artifact names.
        tensorboard_use_gzip (bool, optional):
            Whether to use gzip for the trace. Defaults to False.
        record_shapes (bool, optional): Whether to record tensor shapes.
            Defaults to False.
        profile_memory (bool, optional): Whether to profile memory.
            Defaults to True.
        with_stack (bool, optional): Whether to record stack info.
            Defaults to False.
        with_flops (bool, optional): Whether to estimate flops for operators.
            Defaults to True.
    """

    def __init__(
        self,
        folder_format: str = '{run_name}/torch_profiler_traces',
        filename_format: str = 'ep{epoch}-ba{batch}-rank{rank}.json',
        artifact_name_format: str = '{run_name}/torch_profiler_traces/ep{epoch}-ba{batch}-rank{rank}.json',
        *,
        overwrite: bool = False,
        use_gzip: bool = False,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        num_traces_to_keep: int = -1,
    ) -> None:
        self.overwrite = overwrite
        self.use_gzip = use_gzip
        self.folder_format = folder_format
        self.filename_format = filename_format
        self.artifact_name_format = artifact_name_format
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.num_traces_to_keep = num_traces_to_keep
        self.profiler: Optional[torch.profiler.profile] = None

    def init(self, state: State, logger: Logger) -> None:
        if state.profiler is None:
            raise RuntimeError(
                textwrap.dedent("""\
                    To use the dataloader profiler, state.profiler must be set.
                    Make sure to run composer with the profiler -- i.e. with the `--profiler` CLI flag."""))

        folder_name = format_name_with_dist(self.folder_format, logger.run_name)
        os.makedirs(folder_name, exist_ok=True)
        if not self.overwrite:
            ensure_folder_is_empty(folder_name)

        dist.barrier()

        def scheduler_fn(profiler_step: int) -> TorchProfilerAction:
            # Invoked on every batch, at the batch end
            # The `profiler_step` will be 1 ahead of what was just profiled, but because
            # The state.timer is also incremented before batch_end, these values should be consistent
            # Wrapping the default scheduling function to deal with epoch boundaries
            # Giving the torch scheduler the batch in the epoch, not the global step

            next_batch_in_epoch = int(state.timer.batch_in_epoch)
            if profiler_step == 0:
                next_batch_in_epoch = 0
            assert state.profiler is not None, "composer profiler should be defined"
            composer_profiler_action = state.profiler.get_action(next_batch_in_epoch)
            next_composer_profiler_action = state.profiler.get_action(next_batch_in_epoch + 1)
            if next_batch_in_epoch == state.steps_per_epoch:
                if composer_profiler_action == ProfilerAction.ACTIVE:
                    # force saving at epoch boundaries
                    return TorchProfilerAction.RECORD_AND_SAVE
            if composer_profiler_action == ProfilerAction.ACTIVE and next_composer_profiler_action != ProfilerAction.ACTIVE:
                return TorchProfilerAction.RECORD_AND_SAVE
            if composer_profiler_action == ProfilerAction.ACTIVE:
                return TorchProfilerAction.RECORD
            if composer_profiler_action == ProfilerAction.WARMUP:
                return TorchProfilerAction.WARMUP
            assert composer_profiler_action == ProfilerAction.SKIP, "invariant error"
            return TorchProfilerAction.NONE

        def handler_fn(prof: torch.profiler.profiler.profile):

            assert state.profiler is not None

            timestamp = state.timer.get_timestamp()

            trace_file_name = os.path.join(
                folder_name,
                format_name_with_dist_and_time(self.filename_format, run_name=logger.run_name, timestamp=timestamp),
            )

            prof.export_chrome_trace(trace_file_name)

            state.profiler.record_chrome_json_trace_file(trace_file_name)

            trace_artifact_name = format_name_with_dist_and_time(self.artifact_name_format,
                                                                 run_name=logger.run_name,
                                                                 timestamp=timestamp)

            logger.file_artifact(LogLevel.BATCH,
                                 artifact_name=trace_artifact_name,
                                 file_path=trace_file_name,
                                 overwrite=self.overwrite)

        self.profiler = torch.profiler.profile(
            schedule=scheduler_fn,
            on_trace_ready=handler_fn,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )
        self.profiler.__enter__()

    def batch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        assert self.profiler is not None, _PROFILE_MISSING_ERROR
        self.profiler.add_metadata_json("global_rank", json.dumps(dist.get_global_rank()))
        self.profiler.step()

    def batch_start(self, state: State, logger: Logger) -> None:
        del state  # unused
        assert self.profiler is not None, _PROFILE_MISSING_ERROR
        logger.data_batch({"profiler/state": self.profiler.current_action.name})

    def close(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
