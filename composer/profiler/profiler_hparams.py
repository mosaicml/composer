# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter classes for the :mod:`~composer.profiler`.

Attributes:
    trace_handler_registry: Trace handler registry.
    profiler_scheduler_registry: Profiler scheduler registry.
"""

from __future__ import annotations

from typing import Callable, Dict, Type, Union

import yahp as hp

from composer.core.state import State
from composer.profiler.json_trace_handler import JSONTraceHandler
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.profiler_schedule import cyclic_schedule
from composer.profiler.trace_handler import TraceHandler

__all__ = [
    'trace_handler_registry',
    'profiler_scheduler_registry',
]

trace_handler_registry: Dict[str, Union[Type[TraceHandler], Type[hp.Hparams]]] = {
    'json': JSONTraceHandler,
}

ProfilerScheduler = Callable[[State], ProfilerAction]

profiler_scheduler_registry: Dict[str, Union[Callable[..., ProfilerScheduler], Type[hp.Hparams],]] = {
    'cyclic': cyclic_schedule,
}
