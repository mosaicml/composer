# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, List

import yahp as hp

from composer.callbacks import CallbackHparams
from composer.profiler import Profiler, ProfilerEventHandler

if TYPE_CHECKING:
    from composer.profiler.json_trace import JSONTraceHandler


@dataclasses.dataclass
class ProfilerCallbackHparams(CallbackHparams, abc.ABC):
    pass


@dataclasses.dataclass
class ProfilerEventHandlerHparams(hp.Hparams, abc.ABC):
    """Base class for profile event handler hparams."""

    @abc.abstractmethod
    def initialize_object(self) -> ProfilerEventHandler:
        """Constructs and returns an instance of the :class:`ProfilerEventHandler`.

        Returns:
            ProfilerEventHandler: The event handler.
        """
        pass


@dataclasses.dataclass
class JSONTraceHandlerHparams(ProfilerEventHandlerHparams):
    """:class:`~composer.profiler.json_trace.JSONTraceHandler` hyperparameters.

    See :class:`~composer.profiler.json_trace.JSONTraceHandler` for documentation."""

    flush_every_n_batches: int = hp.optional("Interval at which to flush the logfile.", default=100)
    buffering: int = hp.optional("Buffering parameter passed to :meth:`open` when opening the logfile.", default=-1)
    output_directory: str = hp.optional("Directory, relative to the run directory, to store traces.",
                                        default="composer_profiler")

    def initialize_object(self) -> JSONTraceHandler:
        from composer.profiler.json_trace import JSONTraceHandler
        return JSONTraceHandler(**dataclasses.asdict(self))


@dataclasses.dataclass
class ProfilerHparams(hp.Hparams):
    """Parameters for the :class:`~composer.core.profiler.Profiler`.

    Parameters:
        trace_event_handlers (List[ProfilerEventHandlerHparams], optional):
            List of prameters for the trace event handlers. (Default: [:class:`JSONTraceHandlerHparams`])
        profilers (List[ProfilerCallbackHparams]): List of :class:`ProfilerCallbackHparams` to use.
            (Default: :class:`DataloaderProfilerHparams`, :class:`SystemProfilerHparams`, and :class:`TorchProfilerHparams`)
        skip_first (int, optional): Number of batches to skip profiling at epoch start. (Default: ``0``)
        wait (int, optional): For each profiling cycle, number of batches to skip at the beginning of the cycle. (Default: ``0``)
        warmup (int, optional): For each profiling cycle, number of batches to be in the warmup state
            after skipping ``wait`` batches.. (Default: ``1``)
        active (int, optional): For each profiling cycle, number of batches to record after warming up. (Default: ``4``)
        repeat (int, optional): Number of profiling cycles to perform per epoch. Set to ``0`` to record the entire epoch. (Default: ``1``)
    """
    hparams_registry = {"trace_event_handlers": {"json": JSONTraceHandlerHparams}}

    trace_event_handlers: List[ProfilerEventHandlerHparams] = hp.optional(
        "Trace event handler hparams", default_factory=lambda: [JSONTraceHandlerHparams()])

    # Composer Trainer profiler
    skip_first: int = hp.optional("Number of batches to skip at epoch start", default=0)
    wait: int = hp.optional("Number of batches to skip at the beginning of each cycle", default=0)
    warmup: int = hp.optional("Number of warmup batches in a cycle", default=1)
    active: int = hp.optional("Number of batches to profile in a cycle", default=4)
    repeat: int = hp.optional("Maximum number of profiling cycle repetitions per epoch (0 for no maximum)", default=1)

    # System profiler
    profile_cpu: bool = hp.optional("Whether to record cpu statistics", default=True)
    profile_sys_memory: bool = hp.optional("Whether to record memory statistics", default=False)
    profile_disk: bool = hp.optional("Whether to record disk statistics", default=False)
    profile_net: bool = hp.optional("Whether to record network statistics", default=False)
    stats_thread_interval_seconds: float = hp.optional("Interval to record stats, in seconds.", default=0.5)

    # Torch profiler
    torch_profiling: bool = hp.optional("Enable Torch profiler", default=False)
    tensorboard_trace_handler_dir: str = hp.optional(
        "directory to store trace results. Relative to the run directory, if set.", default="torch_profiler")
    tensorboard_use_gzip: bool = hp.optional("Whether to use gzip for trace", default=False)

    record_shapes: bool = hp.optional(doc="Whether to record tensor shapes", default=False)
    profile_memory: bool = hp.optional(doc="track tensor memory allocations and frees", default=True)
    with_stack: bool = hp.optional(doc="record stack info", default=False)
    with_flops: bool = hp.optional(doc="estimate flops for operators", default=True)

    def initialize_object(self) -> Profiler:
        return Profiler(event_handlers=[x.initialize_object() for x in self.trace_event_handlers],
                        skip_first=self.skip_first,
                        wait=self.wait,
                        warmup=self.warmup,
                        active=self.active,
                        repeat=self.repeat,
                        profile_cpu=self.profile_cpu,
                        profile_sys_memory=self.profile_sys_memory,
                        profile_disk=self.profile_disk,
                        profile_net=self.profile_net,
                        stats_thread_interval_seconds=self.stats_thread_interval_seconds,
                        torch_profiling=self.torch_profiling,
                        tensorboard_trace_handler_dir=self.tensorboard_trace_handler_dir,
                        tensorboard_use_gzip=self.tensorboard_use_gzip,
                        record_shapes=self.record_shapes,
                        profile_memory=self.profile_memory,
                        with_stack=self.with_stack,
                        with_flops=self.with_flops)
