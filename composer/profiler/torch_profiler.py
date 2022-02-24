# Copyright 2021 MosaicML. All Rights Reserved.

"""Profiler to collect :mod:`torch` performance metrics during training."""

from __future__ import annotations

import functools
import json
import os
import textwrap
from typing import Optional

import torch.profiler
from torch.profiler.profiler import ProfilerAction as TorchProfilerAction

from composer.core import Callback, Logger, State
from composer.profiler._profiler_action import ProfilerAction
from composer.utils import dist, run_directory

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

    To view profiling results, run:

    .. code-block::

        pip install tensorbaord torch_tb_profiler
        tensorboard --logdir tensorboard_trace_handler_dir

    .. note::

        See :doc:`profiler` for additional usage details on :class:`torch.profiler.profile`\\.

    .. note::

        Enabling shape and stack tracing results in additional overhead.
        When ``record_shapes=True`` is specified, the profiler will temporarily hold references to tensors which
        may prevent certain optimizations that depend on the reference count and can introduce extra tensor copies.

    Args:
        tensorboard_trace_handler_dir (str): Directory to store trace results.
            Relative to the run_directory. Defaults to ``torch_profiler`` in the
            run directory.
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
        tensorboard_trace_handler_dir: str = "torch_profiler",
        tensorboard_use_gzip: bool = False,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
    ) -> None:
        super().__init__()
        self.tensorboard_trace_handler_dir = os.path.join(run_directory.get_run_directory(),
                                                          tensorboard_trace_handler_dir)
        self.tensorboard_use_gzip = tensorboard_use_gzip
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.profiler: Optional[torch.profiler.profile] = None

    def _scheduler_fn(self, profiler_step: int, state: State) -> TorchProfilerAction:
        # Invoked on every batch, at the batch end
        # But, it's called one batch in advance.
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

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        assert self.profiler is None, "The profiler should be None upon init"
        if state.profiler is None:
            raise RuntimeError(
                textwrap.dedent("""\
                    To use the dataloader profiler, state.profiler must be set.
                    Make sure to run composer with the profiler -- i.e. with the `--profiler` CLI flag."""))
        self.profiler = torch.profiler.profile(
            schedule=functools.partial(self._scheduler_fn, state=state),
            # TODO(ravi): Instruct the pytorch profiler to dump trace events through our profiler,
            # rather than to a seperate JSON file. Then, temove the tensorboard_trace_handler_dir
            # and tensorboard_use_gzip hparams, and the JSONTraceMerger can be invoked on the
            # close() call of the JSONTraceHandler.
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=self.tensorboard_trace_handler_dir,
                worker_name=f"torch_profiler_{dist.get_global_rank()}",
                use_gzip=self.tensorboard_use_gzip,
            ),
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
        logger.metric_batch({"profiler/state": self.profiler.current_action.name})

    def close(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
