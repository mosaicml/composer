# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Profiler Marker."""

from __future__ import annotations

import functools
import time
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

from composer.profiler.trace_handler import TraceHandler

if TYPE_CHECKING:
    from composer.core import State, Timestamp

__all__ = ['Marker']


class Marker:
    """Profiler Marker.

    Used by the :class:`.Engine` to measure the duration of :class:`.Event` during training.

    .. note::

        :class:`.Marker` should not be instantiated directly; instead use :meth:`.Profiler.marker`.

    Markers can record the following types of events:

    #. Duration: Records the start and stop time of an event of interest (:meth:`.Marker.start()`, :meth:`.Marker.finish()`).
    #. Instant: Record time a particular event occurs, but not the full duration (:meth:`.Marker.instant()`).
    #. Counter: The value of a variable at given time (:meth:`.Marker.counter()`).

    A :class:`.Marker` can also be used as a context manager or decorator to record a duration:

    #. Use a :class:`.Marker` with a context manager:

        .. testsetup::

            from composer.profiler import Profiler, cyclic_schedule
            profiler = Profiler(schedule=cyclic_schedule(), trace_handlers=[], torch_prof_memory_filename=None)
            profiler.bind_to_state(state)

        .. doctest::

            >>> def something_to_measure():
            ...     print("something_to_measure")
            >>> marker = profiler.marker("foo")
            >>> with marker:
            ...     something_to_measure()
            something_to_measure

    #. Use a :class:`.Marker` as a decorator:

        .. testsetup::

            from composer.profiler import Profiler, cyclic_schedule
            profiler = Profiler(schedule=cyclic_schedule(), trace_handlers=[], torch_prof_memory_filename=None)
            profiler.bind_to_state(state)

        .. doctest::

            >>> marker = profiler.marker("foo")
            >>> @marker
            ... def something_to_measure():
            ...     print("something_to_measure")
            >>> something_to_measure()
            something_to_measure
    """

    def __init__(
        self,
        state: State,
        should_record: Callable[[State], bool],
        trace_handlers: Sequence[TraceHandler],
        name: str,
        record_instant_on_start: bool,
        record_instant_on_finish: bool,
        categories: Union[list[str], tuple[str, ...]],
    ) -> None:
        self.state = state
        self.trace_handlers = trace_handlers
        self.name = name
        self.categories = categories
        self.record_instant_on_start = record_instant_on_start
        self.record_instant_on_finish = record_instant_on_finish
        self.should_record = should_record
        self._started = False
        self._recorded_start = False

    def _record_duration_event(self, is_start: bool, wall_clock_time_ns: int, timestamp: Timestamp):
        """Record a duration event."""
        for handler in self.trace_handlers:
            handler.process_duration_event(
                name=self.name,
                categories=self.categories,
                timestamp=timestamp,
                is_start=is_start,
                wall_clock_time_ns=wall_clock_time_ns,
            )

    def _record_instant_event(self, wall_clock_time_ns: int, timestamp: Timestamp):
        """Record an instant event."""
        for handler in self.trace_handlers:
            handler.process_instant_event(
                name=self.name,
                categories=self.categories,
                timestamp=timestamp,
                wall_clock_time_ns=wall_clock_time_ns,
            )

    def _record_counter_event(
        self,
        wall_clock_time_ns: int,
        timestamp: Timestamp,
        values: dict[str, Union[int, float]],
    ) -> None:
        """Record a counter invent."""
        for handler in self.trace_handlers:
            handler.process_counter_event(
                name=self.name,
                categories=self.categories,
                wall_clock_time_ns=wall_clock_time_ns,
                timestamp=timestamp,
                values=values,
            )

    def start(self) -> None:
        """Record the start of a duration event.

        To record the duration of an event, invoke :meth:`.Marker.start` followed by :meth:`.Marker.finish`.

        .. testsetup::

            from composer.profiler import Profiler, cyclic_schedule
            profiler = Profiler(schedule=cyclic_schedule(), trace_handlers=[], torch_prof_memory_filename=None)
            profiler.bind_to_state(state)

        .. doctest::

            >>> def something_to_measure():
            ...     print("something_to_measure")
            >>> marker = profiler.marker("foo")
            >>> marker.start()
            >>> something_to_measure()
            something_to_measure
            >>> marker.finish()
        """
        if self._started:
            raise RuntimeError(
                f'Attempted to start profiler event {self.name}; however, this marker is already started',
            )

        self._recorded_start = self.should_record(self.state)
        if self._recorded_start:
            wall_clock_time = time.time_ns()
            self._record_duration_event(
                is_start=True,
                wall_clock_time_ns=wall_clock_time,
                timestamp=self.state.timestamp,
            )
            if self.record_instant_on_start:
                self._record_instant_event(
                    timestamp=self.state.timestamp,
                    wall_clock_time_ns=wall_clock_time,
                )
        self._started = True

    def finish(self) -> None:
        """Record the end of a duration event.

        See :meth:`.Marker.start()` for a usage example.
        """
        if not self._started:
            raise RuntimeError(
                f'Attempted to finish profiler event {self.name}; however, this profiler event is not yet started',
            )

        wall_clock_time = time.time_ns()
        self._record_duration_event(
            is_start=False,
            timestamp=self.state.timestamp,
            wall_clock_time_ns=wall_clock_time,
        )
        if self.record_instant_on_finish:
            self._record_instant_event(
                wall_clock_time_ns=wall_clock_time,
                timestamp=self.state.timestamp,
            )

        self._started = False

    def instant(self) -> None:
        """Record an instant event.

        To record an instant event:

        .. testsetup::

            from composer.profiler import Profiler, cyclic_schedule
            profiler = Profiler(schedule=cyclic_schedule(), trace_handlers=[], torch_prof_memory_filename=None)
            profiler.bind_to_state(state)

        .. doctest::

            >>> def something_to_measure():
            ...     print("something_to_measure")
            >>> marker = profiler.marker("instant")
            >>> marker.instant()
            >>> something_to_measure()
            something_to_measure
        """
        if self.should_record(self.state):
            self._record_instant_event(
                wall_clock_time_ns=time.time_ns(),
                timestamp=self.state.timestamp,
            )

    def counter(self, values: dict[str, Union[float, int]]) -> None:
        """Record a counter event.

        To record a counter event:

        .. testsetup::

            from composer.profiler import Profiler, cyclic_schedule
            profiler = Profiler(schedule=cyclic_schedule(), trace_handlers=[], torch_prof_memory_filename=None)
            profiler.bind_to_state(state)

        .. doctest::

            >>> marker = profiler.marker("foo")
            >>> counter_event = 5
            >>> marker.counter({"counter_event": counter_event})
            >>> counter_event = 10
            >>> marker.counter({"counter_event": counter_event})
        """
        if self.should_record(self.state):
            self._record_counter_event(
                wall_clock_time_ns=time.time_ns(),
                values=values,
                timestamp=self.state.timestamp,
            )

    def __enter__(self) -> Marker:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        del exc_type, exc, traceback  # unused
        self.finish()

    def __call__(self, func: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        if func is None:
            # for decorators of the style @Marker(),
            # return self so it's equivalent to @Marker
            return self

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any):
            with self:
                func(*args, **kwargs)

        return wrapped
