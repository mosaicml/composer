# Copyright 2021 MosaicML. All Rights Reserved.

"""Example usage and definition of hparams."""

from __future__ import annotations

import abc
import dataclasses
from typing import Callable, Optional

import yahp as hp

from composer.core import State
from composer.profiler._profiler import cyclic_schedule
from composer.profiler._profiler_action import ProfilerAction
from composer.profiler._trace_handler import TraceHandler
from composer.profiler.json_trace_handler import JSONTraceHandler

__all__ = [
    "TraceHandlerHparams", "JSONTraceHparams", "trace_handler_registory", "ProfileScheduleHparams",
    "CyclicProfilerScheduleHparams", "profiler_scheduler_registry"
]


@dataclasses.dataclass
class TraceHandlerHparams(hp.Hparams, abc.ABC):
    """Base class for profiler trace destination hparams."""

    @abc.abstractmethod
    def initialize_object(self) -> TraceHandler:
        """Constructs and returns an instance of the :class:`.TraceHandler`.

        Returns:
            TraceHandler: The trace destination.
        """
        pass


@dataclasses.dataclass
class JSONTraceHparams(TraceHandlerHparams):
    """:class:`.JSONTraceHandler` hyperparameters.

    See :class:`.JSONTraceHandler` for documentation.
    
    Example usage with :class:`.TrainerHparams`\\:

    .. code-block:: yaml

        prof_trace_handlers:
            - json:
                folder_format: '{run_name}/traces'

    """
    folder_format: str = hp.optional("Folder format", default='{run_name}/traces')
    filename_format: str = hp.optional("Filename format string for the profile trace.",
                                       default='ep{epoch}-ba{batch}-rank{rank}.json')
    artifact_name_format: Optional[str] = hp.optional("Artifact name format string for the profiler trace.",
                                                      default='{run_name}/traces/ep{epoch}-ba{batch}-rank{rank}.json')
    merged_trace_filename_format: Optional[str] = hp.optional("Merged trace filename format",
                                                              default='node{node_rank}.json')
    merged_trace_artifact_name_format: Optional[str] = hp.optional("Merged trace file artifact name format",
                                                                   default='{run_name}/traces/merged_trace.json')
    overwrite: bool = hp.optional("Overwrite", default=False)
    num_trace_cycles_to_keep: int = hp.optional("Num trace files to keep", default=-1)

    def initialize_object(self) -> JSONTraceHandler:
        return JSONTraceHandler(**dataclasses.asdict(self))


trace_handler_registory = {"json": JSONTraceHparams}


@dataclasses.dataclass
class ProfileScheduleHparams(hp.Hparams, abc.ABC):
    """Profiler schedule hparams."""

    @abc.abstractmethod
    def initialize_object(self) -> Callable[[State], ProfilerAction]:
        """Constructs and returns a profiler scheduler.

        The scheduler is passed as the ``schedule`` parameter into the :class:`~composer.profiler.Profiler`.

        Returns:
            (state) -> ProfilerAction: The profiler scheduler.
        """
        pass


@dataclasses.dataclass
class CyclicProfilerScheduleHparams(ProfileScheduleHparams):
    skip_first: int = hp.optional("skip first", default=0)
    wait: int = hp.optional("wait", default=0)
    warmup: int = hp.optional("warmup", default=1)
    active: int = hp.optional("active", default=4)
    repeat: int = hp.optional("repeat", default=1)

    def initialize_object(self) -> Callable[[State], ProfilerAction]:
        return cyclic_schedule(**dataclasses.asdict(self))


profiler_scheduler_registry = {'cyclic': CyclicProfilerScheduleHparams}
