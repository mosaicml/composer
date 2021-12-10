# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
import dataclasses
import json
import os
import threading
import time
from functools import wraps
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, TYPE_CHECKING

import yahp as hp

from composer.core.callback import Callback
from composer.core.state import State
from composer.utils import ddp
from composer.utils.run_directory import get_relative_to_run_directory
from composer.utils.string_enum import StringEnum

if TYPE_CHECKING:
    from composer.core.state import State


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


class ProfilerEventHandler(Callback, abc.ABC):
    """Base class for profiler event handlers.

    Subclasses should implement :meth:`process_duration_event` and
    :meth:`process_instant_event`. These methods are invoked by the :class:`MosaicProfiler`
    whenever there is an event to record.

    Since :class:`ProfilerEventHandler` subclasses :class:`~composer.Callback`,
    event handlers can run on :class:`~composer.Event`\\s (such as on :attr:`~composer.Event.INIT` to open files or on
    :attr:`~composer.Event.BATCH_END` to periodically dump data to files) and use :meth:`~composer.Callback.close`
    to perform any cleanup.
    """

    def process_duration_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        epoch: int,
        step: int,
        wall_clock_time_ns: int,
        process_id: int,
        thread_id: int,
    ) -> None:
        """Called by the :class:`MosaicProfiler` whenever there is a duration event to record.

        This method is called twice for each duration event -- once with ``is_start = True``,
        and then again with ``is_start = False``. Interleaving events are not permitted.
        Specifically, for each event (identified by the ``name``), a call with ``is_start = True`` will be followed
        by a call with ``is_start = False`` before another call with ``is_start = True``.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            is_start (bool): Whether the event is a start event or end event.
            epoch (int): The epoch corresponding to the event.
            step (int): The step corresponding to the event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            process_id (int): The process id corresponding to the event.
            thread_id (int): The thread id corresponding to the event.
        """
        del name, categories, is_start, epoch, step, wall_clock_time_ns, process_id, thread_id  # unused
        pass

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        epoch: int,
        step: int,
        wall_clock_time_ns: int,
        process_id: int,
        thread_id: int,
    ) -> None:
        """Called by the :class:`MosaicProfiler` whenever there is an instant event to record.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            is_start (bool): Whether the event is a start event or end event.
            epoch (int): The epoch corresponding to the event.
            step (int): The step corresponding to the event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            process_id (int): The process id corresponding to the event.
            thread_id (int): The thread id corresponding to the event.
        """
        del name, categories, epoch, step, wall_clock_time_ns, process_id, thread_id  # unused
        pass

    def process_counter_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        wall_clock_time_ns: int,
        process_id: int,
        thread_id: int,
        values: Dict[str, Union[int, float]],
    ) -> None:
        """Called by the :class:`MosaicProfiler` whenever there is an counter event to record.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            process_id (int): The process id corresponding to the event.
            thread_id (int): The thread id corresponding to the event.
            values (Dict[str, int | float]): The values corresponding to this counter event
        """
        del name, categories, wall_clock_time_ns, process_id, thread_id, values  # unused
        pass


class MosaicProfilerAction(StringEnum):
    """Action states for the :class:`MosaicProfiler`.

    Attributes:
        SKIP: Not currently recording new events at the batch level or below.
            However, any open duration events will still be closed.
        WARMUP: The profiler 
        ACTIVE: Record all events. 
    """
    SKIP = "skip"
    WARMUP = "warmup"
    ACTIVE = "active"


class MosaicProfiler:
    """The MosaicProfiler produces a trace of the training graph.

    Specifically, it records:

    #. The duration of each section of the training loop, such as the time it takes to perform a forward pass, backward pass, batch, epoch, etc...

    #. The latency each algorithm and callback adds when executing on each event.

    #. The latency it takes for the dataloader to yield a batch.
    
    The ``event_handlers`` then record and save this data to a usable trace. 

    Args:
        state (State): The state.
        event_handlers (Sequence[ProfilerEventHandler]): Event handlers which record and save profiling data to traces.
        skip_first (int, optional): Number of batches to skip profiling at epoch start. (Default: ``0``)
        wait (int, optional): For each profiling cycle, number of batches to skip at the beginning of the cycle. (Default: ``0``)
        warmup (int, optional): For each profiling cycle, number of batches to be in the warmup state
            after skipping ``wait`` batches.. (Default: ``1``)
        active (int, optional): For each profiling cycle, number of batches to record after warming up. (Default: ``4``)
        repeat (int, optional): Number of profiling cycles to perform per epoch. Set to ``0`` to record the entire epoch. (Default: ``1``)
    """

    def __init__(
        self,
        state: State,
        event_handlers: Sequence[ProfilerEventHandler],
        skip_first: int = 0,
        wait: int = 0,
        warmup: int = 1,
        active: int = 4,
        repeat: int = 1,
    ) -> None:
        self._names_to_markers: Dict[str, Marker] = {}
        self._event_handlers = event_handlers
        self.state = state
        self.skip_first = skip_first
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self._action = MosaicProfilerAction.SKIP

    def get_action(self, batch_idx: int) -> MosaicProfilerAction:
        """Get the current :class:`MosaicProfilerAction` for the profiler, based upon
        the parameters ``skip_first``, ``wait``, ``warmup``, ``active``, and ``repeat``.

        The profiler skips the first ``skip_first`` batches in every epoch. Then, it performs a cycle of
        skipping ``wait`` batches, warming up for ``warmup`` batches, and recording ``active`` batches.
        It repeats this cylce up to ``repeat`` times per epoch (or for the entire epoch, if ``repeat`` is 0).
        This logic repeats every epoch.

        Returns:
            MosaicProfilerAction: The current action.
        """
        # do wait, then warump, then active, up to repeat times per cycle
        cycle_len = self.wait + self.warmup + self.active
        if self.repeat != 0 and batch_idx >= cycle_len * self.repeat:
            # exhausted the repeat
            return MosaicProfilerAction.SKIP
        position_in_cycle = batch_idx % cycle_len
        if position_in_cycle < self.wait:
            return MosaicProfilerAction.SKIP
        if position_in_cycle < self.wait + self.warmup:
            return MosaicProfilerAction.WARMUP
        return MosaicProfilerAction.ACTIVE

    def marker(
        self,
        name: str,
        actions: Sequence[MosaicProfilerAction] = (MosaicProfilerAction.WARMUP, MosaicProfilerAction.ACTIVE),
        record_instant_on_start: bool = False,
        record_instant_on_finish: bool = False,
        categories: Union[List[str], Tuple[str, ...]] = tuple()) -> Marker:
        """Get a :class:`Marker`.

        If a :class:`Marker` with the specified ``name`` does not already exist, it will be created.
        Otherwise, the existing instance will be returned.

        Args:
            name (str): The name for the :class:`Marker`.
            actions (Sequence[MosaicProfilerAction], optional): :class:`MosaicProfilerAction` states to record on.
                By default, markers will record on :attr:`MosaicProfilerAction.WARMUP` and :attr:`MosaicProfilerAction.ACTIVE`
            record_instant_on_start (bool, optional): Whether to record an instant event whenever the marker is started
            record_instant_on_finish (bool, optional): Whether to record an instant event whenever the marker is finished
            categories (List[str] | Tuple[str, ...], optional): Categories for this marker.

        Returns:
            Marker: [description]
        """
        if name not in self._names_to_markers:
            self._names_to_markers[name] = Marker(
                self,
                name,
                actions=actions,
                record_instant_on_start=record_instant_on_start,
                record_instant_on_finish=record_instant_on_finish,
                categories=categories,
            )
        self._names_to_markers[name].categories = categories
        return self._names_to_markers[name]

    def record_duration_event(self, marker: Marker, is_start: bool, wall_clock_time_ns: int, process_id: int,
                              thread_id: int, epoch: int, step: int):
        """Record a duration event.

        .. note::

            This method should not be invoked directly. Instead, create a :class:`Marker`
            via :meth:`marker`, and then invoke :meth:`Marker.start` or :meth:`Marker.finish`.

        Args:
            marker (Marker): The marker.
            is_start (bool): Whether this is the start or end of the duration event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            process_id (int): The process id where the event was triggered
            thread_id (int): The thread id where the event was triggered
            epoch (int): The epoch at which the event was triggered.
            step (int): The step at which the event was triggered.
        """
        for handler in self._event_handlers:
            handler.process_duration_event(
                name=marker.name,
                categories=marker.categories,
                epoch=epoch,
                step=step,
                is_start=is_start,
                wall_clock_time_ns=wall_clock_time_ns,
                process_id=process_id,
                thread_id=thread_id,
            )

    def record_instant_event(self, marker: Marker, wall_clock_time_ns: int, process_id: int, thread_id: int, epoch: int,
                             step: int):
        """Record an instant event.

        .. note::

            This method should not be invoked directly. Instead, create a :class:`Marker`
            via :meth:`marker`, and then invoke :meth:`Marker.instant`.

        Args:
            marker (Marker): The marker.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            process_id (int): The process id where the event was triggered.
            thread_id (int): The thread id where the event was triggered.
            epoch (int): The epoch at which the event was triggered.
            step (int): The step at which the event was triggered.
        """
        for handler in self._event_handlers:
            handler.process_instant_event(
                name=marker.name,
                categories=marker.categories,
                epoch=epoch,
                step=step,
                wall_clock_time_ns=wall_clock_time_ns,
                process_id=process_id,
                thread_id=thread_id,
            )

    def record_counter_event(
        self,
        marker: Marker,
        wall_clock_time_ns: int,
        process_id: int,
        thread_id: int,
        values: Dict[str, Union[int, float]],
    ) -> None:
        """Record a counter invent

        .. note::

            This method should not be invoked directly. Instead, create a :class:`Marker`
            via :meth:`marker`, and then invoke :meth:`Marker.counter`.

        Args:
            marker (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            epoch (Optional[int]): The epoch, if applicable, corresponding to the event.
            step (int): The step, if applicable, corresponding to the event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            process_id (int): The process id corresponding to the event.
            thread_id (int): The thread id corresponding to the event.
            values (Dict[str, int | float]): The values corresponding to this counter event
        """
        for handler in self._event_handlers:
            handler.process_counter_event(
                name=marker.name,
                categories=marker.categories,
                wall_clock_time_ns=wall_clock_time_ns,
                process_id=process_id,
                thread_id=thread_id,
                values=values,
            )

    def _merge_traces(self):
        """Merge trace files in the output directory.

        Compute the clock sync difference across every process and add the difference
        to each recorded trace event.
        """

        from composer.profiler.json_trace import JSONTraceHandler as JSONTraceHandler
        if ddp.get_global_rank() != 0:
            return

        for event_handler in self._event_handlers:
            if isinstance(event_handler, JSONTraceHandler):
                rank_0_metadata_filename = f"rank_0.metadata.trace.json"
                rank_0_metadata_file = get_relative_to_run_directory(
                    os.path.join(event_handler.output_directory, rank_0_metadata_filename))
                with open(rank_0_metadata_file, "r") as f:
                    trace_metadata = json.load(f)
                rank_0_clock_sync = trace_metadata["clock_sync_timestamp_us"]

                merged_trace_file = open(
                    get_relative_to_run_directory(os.path.join(event_handler.output_directory, f"merged_trace.json")),
                    "x")
                merged_trace_file.write("[\n")

                first_line = True
                for ddp_rank in range(0, ddp.get_world_size()):
                    metadata_filename = f"rank_{ddp_rank}.metadata.trace.json"
                    metadata_file = get_relative_to_run_directory(
                        os.path.join(event_handler.output_directory, metadata_filename))
                    with open(metadata_file, "r") as f:
                        trace_metadata = json.load(f)
                    clock_sync = trace_metadata["clock_sync_timestamp_us"]
                    clock_sync_diff = rank_0_clock_sync - clock_sync

                    trace_filename = f"rank_{ddp_rank}.trace.json"
                    trace_file = get_relative_to_run_directory(
                        os.path.join(event_handler.output_directory, trace_filename))
                    with open(trace_file, "r") as f:
                        trace_data = json.load(f)

                    for event in trace_data:
                        if "ts" in event:
                            event["ts"] = event["ts"] + clock_sync_diff
                        event_string = json.dumps(
                            event,
                            separators=(',', ': '),
                        )
                        if not first_line:
                            merged_trace_file.write(",\n")
                        else:
                            first_line = False
                        merged_trace_file.write(event_string)
                merged_trace_file.write("\n]")
                merged_trace_file.close()


class Marker:
    """Record when something happens or how long something takes.

    .. note::

        :class:`Marker` should not be instantiated directly; instead use :meth:`MosaicProfiler.marker`.

    To use a `Marker` to record a duration, you can:

        #. Invoke :meth:`Marker.start` followed by :meth:`Marker.finish`

            .. code-block:: python

                marker = profiler.marker("foo")
                marker.start()
                something_to_measure()
                marker.finish()

        #. Use a :class:`Marker` with a context manager:

            .. code-block:: python

                marker = profiler.marker("foo")
                with marker:
                    something_to_measure()

        #. Use a :class:`Marker` as a decorator:

            .. code-block:: python

                marker = profiler.marker("foo")

                @marker
                def something_to_measure():
                    ...
                
                something_to_measure()

    To use a :class:`Marker` to record an instant, call :meth:`instant`

    Args:
        mosaic_profiler (MosaicProfiler): The profiler.
        name (str): The name of the event.
        actions (Sequence[MosaicProfilerAction], optional): :class:`MosaicProfilerAction` states to record on.
        record_instant_on_start (bool): Whether to record an instant event whenever the marker is started
        record_instant_on_finish (bool): Whether to record an instant event whenever the marker is finished
        categories (List[str] | Tuple[str, ...]]): Categories corresponding to this event.
    """

    def __init__(self, mosaic_profiler: MosaicProfiler, name: str, actions: Sequence[MosaicProfilerAction],
                 record_instant_on_start: bool, record_instant_on_finish: bool,
                 categories: Union[List[str], Tuple[str, ...]]) -> None:

        self._instrumentation = mosaic_profiler
        self.name = name
        self.actions = actions
        self.categories = categories
        self.record_instant_on_start = record_instant_on_start
        self.record_instant_on_finish = record_instant_on_finish
        if name in mosaic_profiler._names_to_markers:
            if mosaic_profiler._names_to_markers[name] is not self:
                raise RuntimeError(
                    f"{self.__class__.__name__} should not be instantiated directly. Instead, use {mosaic_profiler.__class__.__name__}.marker(name)"
                )
        self._started = False
        self._action_at_start = None

    def start(self) -> None:
        """Record the start of a duration event."""
        if self._started:
            raise RuntimeError(
                f"Attempted to start profiler event {self.name}; however, this marker is already started")

        epoch = self._instrumentation.state.epoch
        step = self._instrumentation.state.step
        batch_idx = self._instrumentation.state.batch_idx
        self._action_at_start = self._instrumentation.get_action(batch_idx)
        if self._action_at_start in self.actions:
            wall_clock_time = time.time_ns()
            self._instrumentation.record_duration_event(
                self,
                is_start=True,
                wall_clock_time_ns=wall_clock_time,
                epoch=epoch,
                step=step,
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
            )
            if self.record_instant_on_start:
                self._instrumentation.record_instant_event(
                    self,
                    epoch=epoch,
                    step=step,
                    wall_clock_time_ns=wall_clock_time,
                    process_id=os.getpid(),
                    thread_id=threading.get_ident(),
                )
        self._started = True

    def finish(self) -> None:
        """Record the end of a duration event."""
        if not self._started:
            raise RuntimeError(
                f"Attempted to finish profiler event {self.name}; however, this profiler event is not yet started")

        if self._action_at_start in self.actions:
            wall_clock_time = time.time_ns()
            epoch = self._instrumentation.state.epoch
            step = self._instrumentation.state.step
            self._instrumentation.record_duration_event(
                self,
                is_start=False,
                epoch=epoch,
                step=step,
                wall_clock_time_ns=wall_clock_time,
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
            )
            if self.record_instant_on_finish:
                self._instrumentation.record_instant_event(
                    self,
                    wall_clock_time_ns=wall_clock_time,
                    epoch=epoch,
                    step=step,
                    process_id=os.getpid(),
                    thread_id=threading.get_ident(),
                )
        self._started = False

    def instant(self) -> None:
        """Record an instant event."""
        epoch = self._instrumentation.state.epoch
        step = self._instrumentation.state.step
        batch_idx = self._instrumentation.state.batch_idx
        if self._instrumentation.get_action(batch_idx) in self.actions:
            self._instrumentation.record_instant_event(
                self,
                wall_clock_time_ns=time.time_ns(),
                epoch=epoch,
                step=step,
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
            )

    def counter(self, values: Dict[str, Union[float, int]]) -> None:
        """Record a counter event."""
        batch_idx = self._instrumentation.state.batch_idx
        if self._instrumentation.get_action(batch_idx) in self.actions:
            self._instrumentation.record_counter_event(
                self,
                wall_clock_time_ns=time.time_ns(),
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
                values=values,
            )

    def __enter__(self) -> Marker:
        self.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.finish()

    def __call__(self, func: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
        if func is None:
            # for decorators of the style @Marker(),
            # return self so it's equivalent to @Marker
            return self

        @wraps(func)
        def wrapped(*args, **kwargs):
            with self:
                func(*args, **kwargs)

        return wrapped
