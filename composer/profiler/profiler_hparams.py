# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter classes for the :mod:`~composer.profiler`.

Attributes:
    trace_handler_registry (Dict[str, Type[TraceHandlerHparams]]): Trace handler registry.
    profiler_scheduler_registry (Dict[str, Type[ProfileScheduleHparams]]): Profiler scheduler registry.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Callable, List, Optional, Type, cast

import yahp as hp

from composer.core.state import State
from composer.profiler import Profiler
from composer.profiler.json_trace_handler import JSONTraceHandler
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.profiler_schedule import cyclic_schedule
from composer.profiler.trace_handler import TraceHandler

__all__ = [
    "TraceHandlerHparams",
    "JSONTraceHparams",
    "trace_handler_registry",
    "ProfileScheduleHparams",
    "CyclicProfilerScheduleHparams",
    "profiler_scheduler_registry",
    "ProfilerHparams",
]


@dataclasses.dataclass
class TraceHandlerHparams(hp.Hparams, abc.ABC):
    """Base class for the :class:`.TraceHandler` hparams."""

    @abc.abstractmethod
    def initialize_object(self) -> TraceHandler:
        """Constructs and returns an instance of a :class:`.TraceHandler`.

        Returns:
            TraceHandler: The trace handler.
        """
        pass


@dataclasses.dataclass
class JSONTraceHparams(TraceHandlerHparams):
    """Hyperparameters for the :class:`.JSONTraceHandler`.

    Args:
        folder (str, optional): See :class:`.JSONTraceHandler`.
        filename (str, optional): See :class:`.JSONTraceHandler`.
        artifact_name (str, optional): See :class:`.JSONTraceHandler`.
        merged_trace_filename (str, optional): See :class:`.JSONTraceHandler`.
        merged_trace_artifact_name (str, optional): See :class:`.JSONTraceHandler`.
        overwrite (bool, optional): See :class:`.JSONTraceHandler`.
        num_traces_to_keep (int, optional): See :class:`.JSONTraceHandler`.
    """
    folder: str = hp.optional("Folder format", default='{run_name}/traces')
    filename: str = hp.optional("Filename format string for the profile trace.",
                                default='ep{epoch}-ba{batch}-rank{rank}.json')
    artifact_name: Optional[str] = hp.optional("Artifact name format string for the profiler trace.",
                                               default='{run_name}/traces/ep{epoch}-ba{batch}-rank{rank}.json')
    merged_trace_filename: Optional[str] = hp.optional("Merged trace filename format", default='node{node_rank}.json')
    merged_trace_artifact_name: Optional[str] = hp.optional("Merged trace file artifact name format",
                                                            default='{run_name}/traces/merged_trace.json')
    overwrite: bool = hp.optional("Overwrite", default=False)
    num_traces_to_keep: int = hp.optional("Num trace files to keep", default=-1)

    def initialize_object(self) -> JSONTraceHandler:
        return JSONTraceHandler(**dataclasses.asdict(self))


trace_handler_registry = {"json": JSONTraceHparams}


@dataclasses.dataclass
class ProfileScheduleHparams(hp.Hparams, abc.ABC):
    """Base class for Composer Profiler schedule hparams."""

    @abc.abstractmethod
    def initialize_object(self) -> Callable[[State], ProfilerAction]:
        """Constructs and returns a Composer Profiler scheduler.

        The scheduler is used ``prof_schedule`` argument for the :class:`~composer.trainer.trainer.Trainer`.

        Returns:
            (state) -> ProfilerAction: The profiler scheduler.
        """
        pass


@dataclasses.dataclass
class CyclicProfilerScheduleHparams(ProfileScheduleHparams):
    """Hyperparameters for the :func:`.cyclic_schedule`.

    Args:
        skip_first (int, optional): See :func:`.cyclic_schedule`.
        wait (str, optional): See :func:`.cyclic_schedule`.
        warmup (str, optional): See :func:`.cyclic_schedule`.
        active (str, optional): See :func:`.cyclic_schedule`.
        repeat (str, optional): See :func:`.cyclic_schedule`.
    """
    skip_first: int = hp.optional("skip first", default=0)
    wait: int = hp.optional("wait", default=0)
    warmup: int = hp.optional("warmup", default=1)
    active: int = hp.optional("active", default=4)
    repeat: int = hp.optional("repeat", default=1)

    def initialize_object(self) -> Callable[[State], ProfilerAction]:
        return cyclic_schedule(**dataclasses.asdict(self))


profiler_scheduler_registry = {'cyclic': cast(Type[hp.Hparams], CyclicProfilerScheduleHparams)}


@dataclasses.dataclass
class ProfilerHparams(hp.Hparams):
    """Hyperparameters for the :class:`.Profiler`.

    Args:
        prof_schedule (ProfileScheduleHparams): Profile schedule hparams.
        prof_trace_handlers (List[TraceHandlerHparams]): See :class:`.Profiler`.
        sys_prof_cpu (bool, optional): See :class:`.Profiler`.
        sys_prof_memory (bool, optional): See :class:`.Profiler`.
        sys_prof_disk (bool, optional): See :class:`.Profiler`.
        sys_prof_net (bool, optional): See :class:`.Profiler`.
        sys_prof_stats_thread_interval_seconds (float, optional): See :class:`.Profiler`.
        torch_prof_folder (str, optional): See :class:`~.TorchProfiler`.
        torch_prof_filename (str, optional): See :class:`~.TorchProfiler`.
        torch_prof_artifact_name (str, optional): See :class:`~.TorchProfiler`.
        torch_prof_overwrite (bool, optional): See :class:`~.TorchProfiler`.
        torch_prof_use_gzip (bool, optional): See :class:`~.TorchProfiler`.
        torch_prof_record_shapes (bool, optional): See :class:`~.TorchProfiler`.
        torch_prof_profile_memory (bool, optional): See :class:`~.TorchProfiler`.
        torch_prof_with_stack (bool, optional): See :class:`~.TorchProfiler`.
        torch_prof_with_flops (bool, optional): See :class:`~.TorchProfiler`.
        torch_prof_num_traces_to_keep (int, optional): See :class:`~.TorchProfiler`.
    """

    hparams_registry = {
        "schedule": profiler_scheduler_registry,
    }

    # profiling
    prof_schedule: ProfileScheduleHparams = hp.required("Profile scheduler hparams")
    prof_trace_handlers: List[TraceHandlerHparams] = hp.required("Trace event handlers")

    sys_prof_cpu: bool = hp.optional("Whether to record cpu statistics.", default=True)
    sys_prof_memory: bool = hp.optional("Whether to record memory statistics.", default=False)
    sys_prof_disk: bool = hp.optional("Whether to record disk statistics.", default=False)
    sys_prof_net: bool = hp.optional("Whether to record network statistics.", default=False)
    sys_prof_stats_thread_interval_seconds: float = hp.optional("Interval to record stats, in seconds", default=0.5)

    torch_prof_folder: str = hp.optional('Torch profiler folder format', default='{run_name}/torch_traces')
    torch_prof_filename: str = hp.optional(
        'Torch profiler filename format',
        default='rank{rank}.{batch}.pt.trace.json',
    )
    torch_prof_artifact_name: str = hp.optional(
        'Torch profiler artifact name format',
        default='{run_name}/torch_traces/rank{rank}.{batch}.pt.trace.json',
    )
    torch_prof_overwrite: bool = hp.optional('Torch profiler overwrite', default=False)
    torch_prof_use_gzip: bool = hp.optional('Torch profiler use gzip', default=False)
    torch_prof_record_shapes: bool = hp.optional("Whether to record tensor shapes", default=False)
    torch_prof_profile_memory: bool = hp.optional("Track tensor memory allocations and frees.", default=False)
    torch_prof_with_stack: bool = hp.optional("Record stack information.", default=False)
    torch_prof_with_flops: bool = hp.optional("Estimate flops for operators.", default=False)
    torch_prof_num_traces_to_keep: int = hp.optional('Torch profiler num traces to keep', default=-1)

    def initialize_object(self):

        return Profiler(
            trace_handlers=[x.initialize_object() for x in self.prof_trace_handlers],
            schedule=self.prof_schedule.initialize_object(),

            # System profiler
            sys_prof_cpu=self.sys_prof_cpu,
            sys_prof_memory=self.sys_prof_memory,
            sys_prof_disk=self.sys_prof_disk,
            sys_prof_net=self.sys_prof_net,
            sys_prof_stats_thread_interval_seconds=self.sys_prof_stats_thread_interval_seconds,

            # Torch profiler
            torch_prof_folder=self.torch_prof_folder,
            torch_prof_filename=self.torch_prof_filename,
            torch_prof_artifact_name=self.torch_prof_artifact_name,
            torch_prof_overwrite=self.torch_prof_overwrite,
            torch_prof_use_gzip=self.torch_prof_use_gzip,
            torch_prof_num_traces_to_keep=self.torch_prof_num_traces_to_keep,
            torch_prof_record_shapes=self.torch_prof_record_shapes,
            torch_prof_profile_memory=self.torch_prof_profile_memory,
            torch_prof_with_stack=self.torch_prof_with_flops,
            torch_prof_with_flops=self.torch_prof_with_flops,
        )
