# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses
from typing import List, Optional

import yahp as hp

from composer.core.profiler import ProfilerEventHandlerHparams, MosaicProfiler
from composer.core.state import State
from composer.profiler.json_trace import JSONTraceHandlerHparams


@dataclasses.dataclass
class MosaicProfilerHparams(hp.Hparams):
    """Parameters for the :class:`~composer.core.profiler.MosaicProfiler`.

    Parameters:
        trace_event_handlers (List[ProfilerEventHandlerHparams], optional):
            List of prameters for the trace event handlers. (Default: [:class:`JSONTraceHandlerHparams`])
        skip_first_epoch (bool, optional): Whether to skip profiling the first epoch. (Default: ``False``)
        wait (int): For each profiling cycle, number of batches to skip at the beginning of the cycle. (Default: ``5``)
        active (int): For each profiling cycle, number of batches to record after skipping the ``wait`` batches. (Default: ``5``)
        repeat (Optional[int]): Number of profiling cycles to perform per epoch. Set to ``None`` to record the entire epoch. (Default: ``3``)
    """
    hparams_registry = {"trace_event_handlers": {"json": JSONTraceHandlerHparams}}

    trace_event_handlers: List[ProfilerEventHandlerHparams] = hp.optional(
        "Trace event handler hparams", default_factory=lambda: [JSONTraceHandlerHparams()])

    skip_first_epoch: bool = hp.optional("Whether to skip profiling on the first epoch", default=False)
    wait: int = hp.optional("Number of steps to skip profiling each cycle", default=5)
    active: int = hp.optional("Number of steps to warm up the profiler each cycle", default=5)
    repeat: Optional[int] = hp.optional("Maximum number of cycles per epoch. Set to None to profile the entire epoch",
                                        default=3)

    def initialize_object(self, state: State) -> MosaicProfiler:
        return MosaicProfiler(
            state=state,
            event_handlers=[x.initialize_object() for x in self.trace_event_handlers],
            skip_first_epoch=self.skip_first_epoch,
            wait=self.wait,
            active=self.active,
            repeat=self.repeat,
        )
