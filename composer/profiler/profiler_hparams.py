# Copyright 2021 MosaicML. All Rights Reserved.

"""Example usage and definition of hparams."""

from __future__ import annotations

import abc
import dataclasses
from typing import Optional

import yahp as hp

from composer.profiler._trace_handler import TraceHandler
from composer.profiler.json_trace_handler import JSONTraceHandler

__all__ = ["TraceHandlerHparams", "JSONTraceHparams"]


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

        prof_event_handlers:
            - json:
                flush_every_n_batches: 100
                buffering: -1
                output_directory: profiler_traces

    """
    filename_format: str = hp.optional("Filename format string for the profile trace.",
                                       default='{run_name}/traces/rank_{rank}.json')
    artifact_name_format: Optional[str] = hp.optional("Artifact name format string for the profiler trace.",
                                                      default=None)
    flush_every_n_batches: int = hp.optional("Interval at which to flush the trace file.", default=100)
    buffering: int = hp.optional("Buffering parameter passed to :meth:`open` when opening the trace file.", default=-1)

    def initialize_object(self) -> JSONTraceHandler:
        return JSONTraceHandler(**dataclasses.asdict(self))


trace_handler_registory = {"json": JSONTraceHparams}
