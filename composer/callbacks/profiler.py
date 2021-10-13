from __future__ import annotations

import atexit
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

import torch.profiler
import yahp as hp
from torch.profiler.profiler import ProfilerAction

from composer import Callback
from composer.callbacks.callback_hparams import CallbackHparams
from composer.core.types import StateDict
from composer.utils.ddp import get_global_rank

if TYPE_CHECKING:
    from composer.core import Logger, State

_PROFILE_MISSING_ERROR = "The profiler has not been setup. Please call profiler.training_start() before training starts."


@dataclass
class TorchProfilerHparams(CallbackHparams):
    tensorboard_trace_handler_dir: str = hp.required("directory to store trace results")
    tensorboard_use_gzip: bool = hp.optional("use gzip for trace", default=False)
    record_shapes: bool = hp.optional(doc="record_shapes", default=True)
    profile_memory: bool = hp.optional(doc="track tensor memory allocations and frees", default=False)
    with_stack: bool = hp.optional(doc="record stack info", default=True)
    with_flops: bool = hp.optional(doc="estimate flops for operators", default=True)

    skip: int = hp.optional("Number of batches to skip at epoch start", default=0)
    warmup: int = hp.optional("Number of warmup batches in a cycle", default=1)
    active: int = hp.optional("Number of batches to profile in a cycle", default=5)
    wait: int = hp.optional("Number of batches to skip at the end of each cycle", default=0)

    def initialize_object(self) -> TorchProfiler:
        return TorchProfiler(**asdict(self))


@dataclass
class TorchProfilerState:
    batch_in_epoch: int = 0
    batches_per_epoch: int = 0


class TorchProfiler(Callback):

    def __init__(
        self,
        *,
        skip: int,
        warmup: int,
        active: int,
        wait: int,
        tensorboard_trace_handler_dir: str,
        tensorboard_use_gzip: bool,
        record_shapes: bool,
        profile_memory: bool,
        with_stack: bool,
        with_flops: bool,
    ) -> None:
        super().__init__()
        self.hparams = TorchProfilerHparams(
            skip=skip,
            warmup=warmup,
            active=active,
            wait=wait,
            tensorboard_trace_handler_dir=tensorboard_trace_handler_dir,
            tensorboard_use_gzip=tensorboard_use_gzip,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
        )
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiler_state: TorchProfilerState = TorchProfilerState()
        self._torch_profiler_scheduler = torch.profiler.profiler.schedule(
            wait=self.hparams.wait,
            warmup=self.hparams.warmup,
            active=self.hparams.active,
            skip_first=self.hparams.skip,
        )

    def state_dict(self) -> StateDict:
        return asdict(self.profiler_state)

    def load_state_dict(self, state: StateDict) -> None:
        self.profiler_state = TorchProfilerState(**state)

    def scheduler_fn(self, profiler_step: int) -> ProfilerAction:
        # Invoked on every batch, at the batch end
        # But, it's called one batch in advance.
        # Wrapping the default scheduling function to deal with epoch boundaries
        # Giving the torch scheduler the batch in the epoch, not the global step

        # adding 1 since this is called before the step is incremented
        next_batch_in_epoch = self.profiler_state.batch_in_epoch + 1
        if profiler_step == 0:
            next_batch_in_epoch = 0
        torch_scheduler_action = self._torch_profiler_scheduler(next_batch_in_epoch)
        if next_batch_in_epoch == self.profiler_state.batches_per_epoch:
            if torch_scheduler_action == ProfilerAction.RECORD:
                # force saving at epoch boundaries
                torch_scheduler_action = ProfilerAction.RECORD_AND_SAVE
        return torch_scheduler_action

    def training_start(self, state: State, logger: Logger) -> None:
        assert self.profiler is None, _PROFILE_MISSING_ERROR
        self.profiler = torch.profiler.profile(
            schedule=self.scheduler_fn,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.hparams.tensorboard_trace_handler_dir,
                worker_name=str(get_global_rank()),
                use_gzip=self.hparams.tensorboard_use_gzip,
            ),
            activities=None,  # auto-set
            record_shapes=self.hparams.record_shapes,
            profile_memory=self.hparams.profile_memory,
            with_stack=self.hparams.with_stack,
            with_flops=self.hparams.with_flops,
        )
        self.profiler.__enter__()
        atexit.register(self._close_profiler)

    def batch_end(self, state: State, logger: Logger) -> None:
        assert self.profiler is not None, _PROFILE_MISSING_ERROR
        self.profiler.step()

    def epoch_start(self, state: State, logger: Logger) -> None:
        self.profiler_state.batches_per_epoch = state.steps_per_epoch

    def batch_start(self, state: State, logger: Logger) -> None:
        self.profiler_state.batch_in_epoch = state.batch_idx
        assert self.profiler is not None, _PROFILE_MISSING_ERROR
        logger.metric_batch({"profiler/state": self.profiler.current_action.name})

    def _close_profiler(self) -> None:
        assert self.profiler is not None
        self.profiler.__exit__(None, None, None)
