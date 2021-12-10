# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import functools
import json
import textwrap
import warnings
from typing import Optional

import torch.profiler
from torch.profiler.profiler import ProfilerAction as TorchProfilerAction

from composer.core import Callback, Logger, State
from composer.core.profiler import ProfilerAction
from composer.profiler.profiler_hparams import TorchProfilerHparams
from composer.utils import ddp
from composer.utils.run_directory import get_relative_to_run_directory

_PROFILE_MISSING_ERROR = "The profiler has not been setup. Please call profiler.init() before training starts."


class TorchProfiler(Callback):
    """Profile the execution using :class:`torch.profiler.profile`.

    Profiling results are stored in TensorBoard format in the
    :param tensorboard_trace_handler_dir: folder.

    To view profiling results, run:

    ``tensorboard --logdir tensorboard_trace_handler_dir``

    Also see https://pytorch.org/docs/stable/profiler.html.

    .. note::

        Enabling shape and stack tracing results in additional overhead.
        When ``record_shapes=True`` is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce extra tensor copies.

    Args:
        tensorboard_trace_handler_dir (str): Directory to store trace results.
            Relative to the run_directory. Defaults to `torch_profiler` in the
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
        self.hparams = TorchProfilerHparams(
            tensorboard_trace_handler_dir=get_relative_to_run_directory(tensorboard_trace_handler_dir),
            tensorboard_use_gzip=tensorboard_use_gzip,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
        )
        self.profiler: Optional[torch.profiler.profile] = None
        try:
            import torch_tb_profiler
            del torch_tb_profiler
        except ModuleNotFoundError:
            warnings.warn(
                "TorchTBProfilerNotFound: torch_tb_profiler not found. You will not be able to visualize torch profiler results."
                "To visualize, run `pip install torch-tb-profiler`")

    def _scheduler_fn(self, profiler_step: int, state: State) -> TorchProfilerAction:
        # Invoked on every batch, at the batch end
        # But, it's called one batch in advance.
        # Wrapping the default scheduling function to deal with epoch boundaries
        # Giving the torch scheduler the batch in the epoch, not the global step

        # adding 1 since this is called before the step is incremented

        next_batch_in_epoch = state.batch_idx + 1
        if profiler_step == 0:
            next_batch_in_epoch = 0
        assert state.profiler is not None, "mosaic profiler should be defined"
        mosaic_profiler_action = state.profiler.get_action(next_batch_in_epoch)
        next_mosaic_profiler_action = state.profiler.get_action(next_batch_in_epoch + 1)
        if next_batch_in_epoch == state.steps_per_epoch:
            if mosaic_profiler_action == ProfilerAction.ACTIVE:
                # force saving at epoch boundaries
                return TorchProfilerAction.RECORD_AND_SAVE
        if mosaic_profiler_action == ProfilerAction.ACTIVE and next_mosaic_profiler_action != ProfilerAction.ACTIVE:
            return TorchProfilerAction.RECORD_AND_SAVE
        if mosaic_profiler_action == ProfilerAction.ACTIVE:
            return TorchProfilerAction.RECORD
        if mosaic_profiler_action == ProfilerAction.WARMUP:
            return TorchProfilerAction.WARMUP
        assert mosaic_profiler_action == ProfilerAction.SKIP, "invariant error"
        return TorchProfilerAction.NONE

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        assert self.profiler is None, "The profiler should be None upon init"
        if state.profiler is None:
            raise RuntimeError(
                textwrap.dedent("""To use the dataloader profiler, state.profiler must be set.
                Make sure to run composer with the profiler -- i.e. with the `--profiler` CLI flag."""))
        self.profiler = torch.profiler.profile(
            schedule=functools.partial(self._scheduler_fn, state=state),
            # TODO(ravi): Instruct the pytorch profiler to dump trace events through our profiler,
            # rather than to a seperate JSON file. Then, temove the tensorboard_trace_handler_dir
            # and tensorboard_use_gzip hparams, and the JSONTraceMerger can be invoked on the
            # close() call of the JSONTraceHandler.
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=self.hparams.tensorboard_trace_handler_dir,
                worker_name=f"torch_profiler_{ddp.get_global_rank()}",
                use_gzip=self.hparams.tensorboard_use_gzip,
            ),
            record_shapes=self.hparams.record_shapes,
            profile_memory=self.hparams.profile_memory,
            with_stack=self.hparams.with_stack,
            with_flops=self.hparams.with_flops,
        )
        self.profiler.__enter__()

    def batch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        assert self.profiler is not None, _PROFILE_MISSING_ERROR
        self.profiler.add_metadata_json("global_rank", json.dumps(ddp.get_global_rank()))
        self.profiler.step()

    def batch_start(self, state: State, logger: Logger) -> None:
        del state  # unused
        assert self.profiler is not None, _PROFILE_MISSING_ERROR
        logger.metric_batch({"profiler/state": self.profiler.current_action.name})

    def close(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
