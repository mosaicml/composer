# Copyright 2021 MosaicML. All Rights Reserved.

"""Example usage and definition of hparams."""

from __future__ import annotations

import abc
import dataclasses

import yahp as hp

from composer.profiler import ProfilerEventHandler
from composer.profiler.json_trace import JSONTraceHandler

__all__ = ["ProfilerEventHandler", "JSONTraceHandlerHparams"]


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

    flush_every_n_batches: int = hp.optional("Interval at which to flush the logfile.", default=100)
    buffering: int = hp.optional("Buffering parameter passed to :meth:`open` when opening the logfile.", default=-1)
    output_directory: str = hp.optional("Directory, relative to the run directory, to store traces.",
                                        default="composer_profiler")

    def initialize_object(self) -> JSONTraceHandler:
        return JSONTraceHandler(**dataclasses.asdict(self))
