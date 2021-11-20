# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses
from typing import List

import yahp as hp

from composer.core.profiler import MosaicProfiler, ProfilerEventHandlerHparams
from composer.core.state import State
from composer.profiler.json_trace import JSONTraceHparams


@dataclasses.dataclass
class MosaicProfilerHparams(hp.Hparams):
    hparams_registry = {"trace_event_handlers": {"json": JSONTraceHparams}}
    trace_event_handlers: List[ProfilerEventHandlerHparams] = hp.optional("Trace event handler hparams",
                                                                          default_factory=lambda: [JSONTraceHparams()])

    def initialize_object(self, state: State) -> MosaicProfiler:
        return MosaicProfiler(state, [x.initialize_object() for x in self.trace_event_handlers])
