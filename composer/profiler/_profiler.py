# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import os
import pathlib
import time
from functools import wraps
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from composer.profiler._profiler_action import ProfilerAction
from composer.utils import dist
from composer.utils.iter_helpers import ensure_tuple

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.core.time import Timestamp
    from composer.profiler._trace_handler import TraceHandler

__all__ = ["Marker", "Profiler"]

log = logging.getLogger(__name__)


class Profiler:
    """Records the duration of Trainer :class:`.Event` using the :class:`.Marker` API.

    Specifically, it records:

    #. The duration of each section of the training loop, such as the time it takes to perform a forward pass, backward pass, batch, epoch, etc.

    #. The latency of each algorithm and callback adds when executing on each event.

    The ``event_handlers`` then record and save this data to a trace file.  If no ``event_handlers`` is specified, the
    :class:`.JSONTraceHandler` is used by default.

    .. note::

        The Composer :class:`.Trainer` creates an instance of :class:`.Profiler` when ``merged_trace_file`` is provided.
        The user should not create and directly register an instance of :class:`Profiler` when using the Composer :class:`.Trainer`\\.

    Args:
        state (State): The state.
        event_handlers (Sequence[MarkerHandler]): Event handlers which record and save profiling data to traces.
        skip_first (int, optional): Number of batches to skip profiling at epoch start.  Defaults to ``0``.
        wait (int, optional): For each profiling cycle, number of batches to skip at the beginning of the cycle.
            Defaults to ``0``.
        warmup (int, optional): For each profiling cycle, number of batches to be in the warmup state after skipping ``wait`` batches.
            Defaults to ``1``.
        active (int, optional): For each profiling cycle, number of batches to record after warming up.  Defaults to ``4``.
        repeat (int, optional): Number of profiling cycles to perform per epoch. Set to ``0`` to record the entire epoch.
            Defaults to ``1``.
    """

    def __init__(
        self,
        state: State,
        trace_handlers: Optional[Union[TraceHandler, Sequence[TraceHandler]]] = None,
        skip_first: int = 0,
        wait: int = 0,
        warmup: int = 1,
        active: int = 4,
        repeat: int = 1,
    ) -> None:
        self._names_to_markers: Dict[str, Marker] = {}
        self._trace_handlers = ensure_tuple(trace_handlers)
        self.state = state
        self.skip_first = skip_first
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self._action = ProfilerAction.SKIP

    def get_action(self, batch_idx: int) -> ProfilerAction:
        """Get the current :class:`ProfilerAction` for the profiler, based upon the parameters ``skip_first``, ``wait``,
        ``warmup``, ``active``, and ``repeat``.

        The profiler skips the first ``skip_first`` batches in every epoch. Then, it performs a cycle of
        skipping ``wait`` batches, warming up for ``warmup`` batches, and recording ``active`` batches.
        It repeats this cycle up to ``repeat`` times per epoch (or for the entire epoch, if ``repeat`` is 0).
        This logic repeats every epoch.

        Args:
            batch_idx (int): The index of the current batch.

        Returns:
            ProfilerAction: The current action.
        """
        # do wait, then warump, then active, up to repeat times per cycle
        cycle_len = self.wait + self.warmup + self.active
        if self.repeat != 0 and batch_idx >= cycle_len * self.repeat:
            # exhausted the repeat
            return ProfilerAction.SKIP
        position_in_cycle = batch_idx % cycle_len
        if position_in_cycle < self.wait:
            return ProfilerAction.SKIP
        if position_in_cycle < self.wait + self.warmup:
            return ProfilerAction.WARMUP
        return ProfilerAction.ACTIVE

    @property
    def trace_handlers(self):
        """Profiler trace recorders."""
        return self._trace_handlers

    def collect_trace_file(self, filename: str, format: str):
        # called by the torch profiler and
        ...

    def record_chrome_json_trace_file(self, filepath: Union[str, pathlib.Path]):
        for recorder in self.trace_handlers:
            recorder.process_chrome_json_trace_file(pathlib.Path(filepath))

    def marker(
        self,
        name: str,
        actions: Sequence[ProfilerAction] = (ProfilerAction.WARMUP, ProfilerAction.ACTIVE),
        record_instant_on_start: bool = False,
        record_instant_on_finish: bool = False,
        categories: Union[List[str], Tuple[str, ...]] = tuple()) -> Marker:
        """Create and get an instance of a :class:`Marker`.

        If a :class:`Marker` with the specified ``name`` does not already exist, it will be created.
        Otherwise, the existing instance will be returned.

        For example:

        .. testsetup::

            from composer.profiler import Profiler
            profiler = Profiler(state=state)

        .. doctest::

            >>> marker = profiler.marker("foo")
            >>> marker
            <composer.profiler.Marker object at ...>

        .. note::

            :meth:`Profiler.marker()` should be used to construct markers.  :class:`Marker` **should not** be 
            instantiated directly by the user.

        Please see :meth:`Marker.start()` and :meth:`Marker.finish()` for usage on creating markers to measure duration events,
        :meth:`Marker.instant()` for usage on creating markers to mark instant events and :meth:`Marker.counter()` for usage on
        creating markers for counting.

        Args:
            name (str): The name for the :class:`Marker`.
            actions (Sequence[ProfilerAction], optional): :class:`ProfilerAction` states to record on.
                Defaults to (:attr:`ProfilerAction.WARMUP`, :attr:`ProfilerAction.ACTIVE`).
            record_instant_on_start (bool, optional): Whether to record an instant event whenever the marker is started.
                Defaults to ``False``.
            record_instant_on_finish (bool, optional): Whether to record an instant event whenever the marker is finished.
                Defaults to ``False``.
            categories (Union[List[str], Tuple[str, ...]], optional): Categories for this marker. Defaults to ``None``.

        Returns:
            Marker: Instance of :class:`Marker`.
        """
        if name not in self._names_to_markers:

            def should_record(state: State) -> bool:
                return self.get_action(int(state.timer.batch_in_epoch)) in actions

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


class Marker:
    """Record when something happens or how long something takes.

    Used by the :class:`~composer.core.engine.Engine` to measure the duration of :class:`~composer.core.event.Event` during training.

    .. note::

        :class:`Marker` should not be instantiated directly; instead use :meth:`Profiler.marker`.

    Markers can record the following types of events:

    #. Duration: Records the start and stop time of an event of interest (:meth:`Marker.start()`, :meth:`Marker.finish()`).
    #. Instant: Record time a particular event occurs, but not the full duration (:meth:`Marker.instant()`).
    #. Counter: The value of a variable at given time (:meth:`Marker.counter()`).

    A :class:`Marker` can also be used as a context manager or decorator to record a duration:

    #. Use a :class:`Marker` with a context manager:

        .. testsetup::

            from composer.profiler import Profiler
            profiler = Profiler(state=state)

        .. doctest::

            >>> def something_to_measure():
            ...     print("something_to_measure")
            >>> marker = profiler.marker("foo")
            >>> with marker:
            ...     something_to_measure()
            something_to_measure

    #. Use a :class:`Marker` as a decorator:

        .. testsetup::

            from composer.profiler import Profiler
            profiler = Profiler(state=state)

        .. doctest::

            >>> marker = profiler.marker("foo")
            >>> @marker
            ... def something_to_measure():
            ...     print("something_to_measure")
            >>> something_to_measure()
            something_to_measure
    """

    def __init__(self, state: State, should_record: Callable[[State], bool], trace_handlers: Sequence[TraceHandler],
                 name: str, record_instant_on_start: bool, record_instant_on_finish: bool,
                 categories: Union[List[str], Tuple[str, ...]]) -> None:
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
                global_rank=dist.get_global_rank(),
                pid=os.getpid(),
            )

    def _record_instant_event(self, wall_clock_time_ns: int, timestamp: Timestamp):
        """Record an instant event."""
        for handler in self.trace_handlers:
            handler.process_instant_event(
                name=self.name,
                categories=self.categories,
                timestamp=timestamp,
                wall_clock_time_ns=wall_clock_time_ns,
                global_rank=dist.get_global_rank(),
                pid=os.getpid(),
            )

    def _record_counter_event(self, wall_clock_time_ns: int, values: Dict[str, Union[int, float]]) -> None:
        """Record a counter invent."""
        for handler in self.trace_handlers:
            handler.process_counter_event(
                name=self.name,
                categories=self.categories,
                wall_clock_time_ns=wall_clock_time_ns,
                global_rank=dist.get_global_rank(),
                pid=os.getpid(),
                values=values,
            )

    def start(self) -> None:
        """Record the start of a duration event.

        To record the duration of an event, invoke :meth:`Marker.start` followed by :meth:`Marker.finish`\\:

        .. testsetup::

            from composer.profiler import Profiler
            profiler = Profiler(state=state)

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
                f"Attempted to start profiler event {self.name}; however, this marker is already started")

        self._recorded_start = self.should_record(self.state)
        if self._recorded_start:
            wall_clock_time = time.time_ns()
            self._record_duration_event(
                is_start=True,
                wall_clock_time_ns=wall_clock_time,
                timestamp=self.state.timer.get_timestamp(),
            )
            if self.record_instant_on_start:
                self._record_instant_event(
                    timestamp=self.state.timer.get_timestamp(),
                    wall_clock_time_ns=wall_clock_time,
                )
        self._started = True

    def finish(self) -> None:
        """Record the end of a duration event.

        See :meth:`Marker.start()` for a usage example.
        """
        if not self._started:
            raise RuntimeError(
                f"Attempted to finish profiler event {self.name}; however, this profiler event is not yet started")

        wall_clock_time = time.time_ns()
        self._record_duration_event(
            is_start=False,
            timestamp=self.state.timer.get_timestamp(),
            wall_clock_time_ns=wall_clock_time,
        )
        if self.record_instant_on_finish:
            self._record_instant_event(
                wall_clock_time_ns=wall_clock_time,
                timestamp=self.state.timer.get_timestamp(),
            )

        self._started = False

    def instant(self) -> None:
        """Record an instant event.

        To record an instant event:

        .. testsetup::

            from composer.profiler import Profiler
            profiler = Profiler(state=state)

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
                timestamp=self.state.timer.get_timestamp(),
            )

    def counter(self, values: Dict[str, Union[float, int]]) -> None:
        """Record a counter event.

        To record a counter event:

        .. testsetup::

            from composer.profiler import Profiler
            profiler = Profiler(state=state)

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
            )

    def __enter__(self) -> Marker:
        self.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        del exc_type, exc, traceback  # unused
        self.finish()

    def __call__(self, func: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        if func is None:
            # for decorators of the style @Marker(),
            # return self so it's equivalent to @Marker
            return self

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any):
            with self:
                func(*args, **kwargs)

        return wrapped
