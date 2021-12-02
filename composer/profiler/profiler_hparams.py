# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses
from typing import List, Optional

import yahp as hp

from composer.core.profiler import MosaicProfiler, ProfilerEventHandlerHparams
from composer.core.state import State
from composer.profiler.json_trace import JSONTraceHparams


@dataclasses.dataclass
class MosaicProfilerHparams(hp.Hparams):
    """Parameters for the Mosaic Profiler.

    Parameters:
        Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip
        the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
        then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
        The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
        the cycles will continue until the profiling is finished.
    """
    hparams_registry = {"trace_event_handlers": {"json": JSONTraceHparams}}

    trace_event_handlers: List[ProfilerEventHandlerHparams] = hp.optional("Trace event handler hparams",
                                                                          default_factory=lambda: [JSONTraceHparams()])

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
