# Copyright 2021 MosaicML. All Rights Reserved.

"""Composer Profiler."""

from __future__ import annotations

import logging
import pathlib
from typing import Callable, Dict, List, Sequence, Tuple, Union

from composer.core.state import State
from composer.profiler.marker import Marker
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.trace_handler import TraceHandler
from composer.utils.iter_helpers import ensure_tuple

__all__ = ["Profiler"]

log = logging.getLogger(__name__)


class Profiler:
    """Composer Profiler.

    See the :doc:`Profiling Guide </trainer/performance_tutorials/profiling>` for additional information.

    .. note::

        The :class:`~composer.trainer.trainer.Trainer` creates an instance of this :class:`.Profiler` class when
        ``prof_trace_handlers`` and ``prof_schedule`` are provided.
        When using the Composer :class:`~composer.trainer.trainer.Trainer`, one does not need to directly create an
        instance of this :class:`Profiler` class.

    Args:
        state (State): The training state.
        schedule ((State) -> ProfilerAction): The profiling scheduling function.

            It takes the training state and returns a :class:`.ProfilerAction`.

        trace_handlers (TraceHandler | Sequence[TraceHandler]):
            Trace handlers which record and save profiling data to traces.
    """

    def __init__(
        self,
        state: State,
        schedule: Callable[[State], ProfilerAction],
        trace_handlers: Union[TraceHandler, Sequence[TraceHandler]],
    ) -> None:
        self._names_to_markers: Dict[str, Marker] = {}
        self._trace_handlers = list(ensure_tuple(trace_handlers))
        self.state = state
        self.schedule = schedule

    @property
    def trace_handlers(self):
        """Profiler trace handlers."""
        return self._trace_handlers

    @trace_handlers.setter
    def trace_handlers(self, trace_handlers: Union[TraceHandler, Sequence[TraceHandler]]):
        """Profiler trace handlers."""
        self._trace_handlers[:] = ensure_tuple(trace_handlers)

    def record_chrome_json_trace_file(self, filepath: Union[str, pathlib.Path]):
        """Record trace events in `Chrome JSON format <https://\\
        docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview>`_ in the trace handlers.

        .. note::

            For custom profiling, it is recommended to use :meth:`marker` instead of manually creating a Chrome JSON
            trace file. By default, the Composer Profiler will automatically saving :class:`.Marker` events in Chrome
            JSON format.

            This method exists for external profilers that natively record events in Chrome JSON format (such as the
            :class:`~composer.profiler.torch_profiler.TorchProfiler`). These profilers can use this method to route
            their profiling traces to the Composer profiler :attr:`~trace_handlers` so events from both the Composer
            Profiler and external profilers are recorded in the same trace file.
        """
        for recorder in self.trace_handlers:
            recorder.process_chrome_json_trace_file(pathlib.Path(filepath))

    def marker(
            self,
            name: str,
            actions: Sequence[ProfilerAction] = (ProfilerAction.WARMUP, ProfilerAction.ACTIVE,
                                                 ProfilerAction.ACTIVE_AND_SAVE),
            record_instant_on_start: bool = False,
            record_instant_on_finish: bool = False,
            categories: Union[List[str], Tuple[str, ...]] = tuple(),
    ) -> Marker:
        """Create and get an instance of a :class:`.Marker`.

        If a :class:`.Marker` with the specified ``name`` does not already exist, it will be created.
        Otherwise, the existing instance will be returned.

        .. note::

            :meth:`.Profiler.marker()` should be used to construct markers.  :class:`.Marker` **should not** be 
            instantiated directly by the user.

            For example:

            .. testsetup::

                from composer.profiler import Profiler, cyclic_schedule

                profiler = Profiler(state=state, schedule=cyclic_schedule(), trace_handlers=[])

            .. doctest::

                >>> marker = profiler.marker("foo")
                >>> marker
                <composer.profiler.marker.Marker object at ...>

        Please see :meth:`.Marker.start()` and :meth:`.Marker.finish()` for usage on creating markers to measure duration events,
        :meth:`.Marker.instant()` for usage on creating markers to mark instant events and :meth:`.Marker.counter()` for usage on
        creating markers for counting.

        Args:
            name (str): The name for the :class:`.Marker`.
            actions (Sequence[ProfilerAction], optional): :class:`.ProfilerAction` states to record on.
                Defaults to (:attr:`~.ProfilerAction.WARMUP`, :attr:`~.ProfilerAction.ACTIVE`,
                :attr:`~.ProfilerAction.ACTIVE_AND_SAVE`).
            record_instant_on_start (bool, optional): Whether to record an instant event whenever the marker is started.
                Defaults to ``False``.
            record_instant_on_finish (bool, optional): Whether to record an instant event whenever the marker is finished.
                Defaults to ``False``.
            categories (Union[List[str], Tuple[str, ...]], optional): Categories for this marker. Defaults to ``None``.

        Returns:
            Marker: Marker instance.
        """
        if name not in self._names_to_markers:

            def should_record(state: State) -> bool:
                return self.schedule(state) in actions

            self._names_to_markers[name] = Marker(
                state=self.state,
                trace_handlers=self.trace_handlers,
                name=name,
                should_record=should_record,
                record_instant_on_start=record_instant_on_start,
                record_instant_on_finish=record_instant_on_finish,
                categories=categories,
            )
        self._names_to_markers[name].categories = categories
        return self._names_to_markers[name]
