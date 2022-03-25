# Copyright 2021 MosaicML. All Rights Reserved.

"""Example usage and definition of hparams."""

from __future__ import annotations

import abc
import dataclasses
from typing import Callable, Optional

import yahp as hp

from composer.core.state import State
from composer.profiler.json_trace_handler import JSONTraceHandler
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.profiler_schedule import cyclic_schedule
from composer.profiler.trace_handler import TraceHandler

__all__ = [
    "TraceHandlerHparams", "JSONTraceHparams", "trace_handler_registory", "ProfileScheduleHparams",
    "CyclicProfilerScheduleHparams", "profiler_scheduler_registry"
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


trace_handler_registory = {"json": JSONTraceHparams}
"""Trace handler registry."""


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


profiler_scheduler_registry = {'cyclic': CyclicProfilerScheduleHparams}
"""Profiler scheduler registry."""
