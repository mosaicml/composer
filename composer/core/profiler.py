# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
import dataclasses
import time
from functools import wraps
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import yahp as hp

from composer.core.callback import Callback
from composer.core.state import State
from composer.core.types import Batch, DataLoader
from composer.datasets.dataloader import WrappedDataLoader
from composer.utils.string_enum import StringEnum


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
        epoch: Optional[int],
        step: Optional[int],
        wall_clock_time_ns: int,
        perf_counter_time_ns: int,
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
            epoch (Optional[int]): The epoch, if applicable, corresponding to the event.
            step (int): The step, if applicable, corresponding to the event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            perf_counter_time_ns (int): The :meth:`time.perf_counter_ns` corresponding to the event.
        """
        del name, categories, is_start, epoch, step, wall_clock_time_ns, perf_counter_time_ns  # unused
        pass

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        epoch: Optional[int],
        step: Optional[int],
        wall_clock_time_ns: int,
        perf_counter_time_ns: int,
    ) -> None:
        """Called by the :class:`MosaicProfiler` whenever there is an instant event to record.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            is_start (bool): Whether the event is a start event or end event.
            epoch (Optional[int]): The epoch, if applicable, corresponding to the event.
            step (int): The step, if applicable, corresponding to the event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            perf_counter_time_ns (int): The :meth:`time.perf_counter_ns` corresponding to the event.
        """
        del name, categories, epoch, step, wall_clock_time_ns, perf_counter_time_ns  # unused
        pass


class ProfiledDataLoader(WrappedDataLoader):
    """Wraps a dataloader to record the duration it takes to yield a batch.
    This class should not be instantiated directly.

    Args:
        profiler (MosaicProfiler): The profiler instance.
        dataloader (DataLoader): The dataloader to profile.
        name (str): The name for the dataloader.
    """

    def __init__(self, profiler: MosaicProfiler, dataloader: DataLoader, name: str) -> None:
        super().__init__(dataloader)
        self._mosaic_profiler = profiler
        self._marker = profiler.marker(f"dataloader/{name}", categories=["dataloader"])
        self._iterator: Optional[Iterator[Batch]] = None

    def __iter__(self) -> ProfiledDataLoader:
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> Batch:
        assert self._iterator is not None
        self._marker.start()
        try:
            return next(self._iterator)
        finally:
            self._marker.finish()


class MosaicProfilerAction(StringEnum):
    """Action states for the :class:`MosaicProfiler`.

    Attributes:
        SKIP: Not currently recording new events at the batch level or below.
            However, any open duration events will still be closed.
        ACTIVE: Record all events. 
    """
    SKIP = "skip"
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
        skip_first_epoch (bool, optional): Whether to skip profiling the first epoch. (Default: ``False``)
        wait (int): For each profiling cycle, number of batches to skip at the beginning of the cycle. (Default: ``5``)
        active (int): For each profiling cycle, number of batches to record after skipping the ``wait`` batches. (Default: ``5``)
        repeat (Optional[int]): Number of profiling cycles to perform per epoch. Set to ``None`` to record the entire epoch. (Default: ``3``)
    """

    def __init__(
        self,
        state: State,
        event_handlers: Sequence[ProfilerEventHandler],
        skip_first_epoch: bool = False,
        wait: int = 5,
        active: int = 5,
        repeat: Optional[int] = 3,
    ) -> None:
        self._names_to_markers: Dict[str, Marker] = {}
        self._event_handlers = event_handlers
        self._state = state
        self._state.train_dataloader = self._wrap_dataloaders_with_markers(self._state.train_dataloader, "train")
        self._state.eval_dataloader = self._wrap_dataloaders_with_markers(self._state.eval_dataloader, "eval")
        self._wait = wait
        self._active = active
        self._repeat = repeat
        self._skip_first_epoch = skip_first_epoch
        self._action = MosaicProfilerAction.SKIP

    def get_action(self):
        """Get the current :class:`MosaicProfilerAction` for the profiler, based upon
        the parameters ``skip_first_epoch``, ``wait``, ``active``, and ``repeat``.

        The profiler skips the first epoch if ``skip_first_epoch`` is True and the current epoch is 0. Otherwise,
        for each epoch, it performs a cycle of skipping ``wait`` batches and then recording ``active`` batches.
        It repeats this cylce up to ``repeat`` times per epoch (or for the entire epoch, if ``repeat`` is None).
        This logic repeats every epoch.

        Returns:
            MosaicProfilerAction: The current action.
        """
        if self._state.epoch == 0 and self._skip_first_epoch:
            return MosaicProfilerAction.SKIP
        # do wait, then warump, then active, up to repeat times per cycle
        cycle_len = self._wait + self._active
        if self._repeat is not None and self._state.batch_idx >= cycle_len * self._repeat:
            # exhausted the repeat
            return MosaicProfilerAction.SKIP
        position_in_cycle = self._state.batch_idx % cycle_len
        if position_in_cycle < self._wait:
            return MosaicProfilerAction.SKIP
        return MosaicProfilerAction.ACTIVE

    def marker(self,
               name: str,
               always_record: bool = False,
               record_instant_on_start: bool = False,
               record_instant_on_finish: bool = False,
               categories: Union[List[str], Tuple[str, ...]] = tuple()) -> Marker:
        """Get a :class:`Marker`.

        If a :class:`Marker` with the specified ``name`` does not already exist, it will be created.
        Otherwise, the existing instance will be returned.

        Args:
            name (str): The name for the :class:`Marker`.
            always_record (bool, optional): Whether to always record events assocaited with this marker.
                If True, then the scheduling parameters (e.g. ``skip_first_epoch``, ``wait``, ``active``, and ``repeat``) are ignored.
                Should be sparingly set to ``True``. (Default: ``False``).
            record_instant_on_start (bool, optional): Whether to record an instant event whenever the marker is started
            record_instant_on_finish (bool, optional): Whether to record an instant event whenever the marker is finished
            categories (List[str] | Tuple[str, ...], optional): Categories for this marker.

        Returns:
            Marker: [description]
        """
        if name not in self._names_to_markers:
            self._names_to_markers[name] = Marker(self, name,
            always_record=always_record,
            record_instant_on_start=record_instant_on_start,
            record_instant_on_finish=record_instant_on_finish,
            categories=categories,
            )
        self._names_to_markers[name].categories = categories
        return self._names_to_markers[name]

    def record_duration_event(self, marker: Marker, is_start: bool, wall_clock_time_ns: int, perf_counter_time_ns: int):
        """Record a duration event.

        .. note::

            This method should not be invoked directly. Instead, create a :class:`Marker`
            via :meth:`marker`, and then invoke :meth:`Marker.start` or :meth:`Marker.finish`.

        Args:
            marker (Marker): The profiler event.
            is_start (bool): Whether this is the start or end of the duration event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            perf_counter_time_ns (int): The :meth:`time.perf_counter_ns` corresponding to the event.
        """
        for handler in self._event_handlers:
            handler.process_duration_event(
                name=marker.name,
                categories=marker.categories,
                epoch=self._state.epoch,
                step=self._state.step,
                is_start=is_start,
                wall_clock_time_ns=wall_clock_time_ns,
                perf_counter_time_ns=perf_counter_time_ns,
            )

    def record_instant_event(self, marker: Marker, wall_clock_time_ns: int, perf_counter_time_ns: int):
        """Record an instant event.

        .. note::

            This method should not be invoked directly. Instead, create a :class:`Marker`
            via :meth:`marker`, and then invoke :meth:`Marker.instant`.

        Args:
            marker (Marker): The profiler event.
            wall_clock_time_ns (int): The :meth:`time.time_ns` corresponding to the event.
            perf_counter_time_ns (int): The :meth:`time.perf_counter_ns` corresponding to the event.
        """
        for handler in self._event_handlers:
            handler.process_instant_event(name=marker.name,
                                          categories=marker.categories,
                                          epoch=self._state.epoch,
                                          step=self._state.step,
                                          wall_clock_time_ns=wall_clock_time_ns,
                                          perf_counter_time_ns=perf_counter_time_ns)

    def _wrap_dataloaders_with_markers(self, dataloader: DataLoader, name: str) -> DataLoader:
        return ProfiledDataLoader(self, dataloader, name)


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
        always_record (bool): Whether to always record the event, regardless of the result of
            :meth:`MosaicProfiler.get_action`.
        record_instant_on_start (bool): Whether to record an instant event whenever the marker is started
        record_instant_on_finish (bool): Whether to record an instant event whenever the marker is finished
        categories (List[str] | Tuple[str, ...]], optional): Categories corresponding to this event.
    """

    def __init__(self,
                 mosaic_profiler: MosaicProfiler,
                 name: str,
                 always_record: bool,
                 record_instant_on_start: bool,
                 record_instant_on_finish: bool,
                 categories: Union[List[str], Tuple[str, ...]] = tuple()) -> None:

        self._instrumentation = mosaic_profiler
        self.name = name
        self.always_record = always_record
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
        self._action_at_start = self._instrumentation.get_action()
        if self._action_at_start == MosaicProfilerAction.ACTIVE or self.always_record:
            wall_clock_time = time.time_ns()
            perf_counter_time = time.perf_counter_ns()
            self._instrumentation.record_duration_event(
                self,
                is_start=True,
                wall_clock_time_ns=wall_clock_time,
                perf_counter_time_ns=perf_counter_time,
            )
            if self.record_instant_on_start:
                self._instrumentation.record_instant_event(
                    self,
                    wall_clock_time_ns=wall_clock_time,
                    perf_counter_time_ns=perf_counter_time,
                )
        self._started = True

    def finish(self) -> None:
        """Record the end of a duration event."""
        if not self._started:
            raise RuntimeError(
                f"Attempted to finish profiler event {self.name}; however, this profiler event is not yet started")

        if self._action_at_start == MosaicProfilerAction.ACTIVE or self.always_record:
            wall_clock_time = time.time_ns()
            perf_counter_time = time.perf_counter_ns()
            self._instrumentation.record_duration_event(
                self,
                is_start=False,
                wall_clock_time_ns=wall_clock_time,
                perf_counter_time_ns=perf_counter_time,
            )
            if self.record_instant_on_finish:
                self._instrumentation.record_instant_event(
                    self,
                    wall_clock_time_ns=wall_clock_time,
                    perf_counter_time_ns=perf_counter_time,
                )
        self._started = False

    def instant(self) -> None:
        """Record an instant event."""
        if self._instrumentation.get_action() == MosaicProfilerAction.ACTIVE or self.always_record:
            self._instrumentation.record_instant_event(
                self,
                wall_clock_time_ns=time.time_ns(),
                perf_counter_time_ns=time.perf_counter_ns(),
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
