# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
from typing import Dict, List, Tuple, Union

from composer.core.callback import Callback
from composer.core.time import Timestamp

__all__ = ["ProfilerEventHandler"]


class ProfilerEventHandler(Callback, abc.ABC):
    """Base class for profiler event handlers.

    Event handlers are responsible for logging trace :class:`.Marker`\\s and saving them to a file in a given trace format for viewing.

    Subclasses should implement :meth:`process_duration_event`, :meth:`process_instant_event` and :meth:`process_counter_event`.
    These methods are invoked by the :class:`.Profiler` whenever there is an event to record.

    Since :class:`ProfilerEventHandler` subclasses :class:`~composer.core.callback.Callback`,
    event handlers can run on :class:`.Event`\\s (such as on :attr:`.Event.INIT` to open files or on
    :attr:`.Event.BATCH_END` to periodically dump data to files) and use :meth:`.Callback.close`
    to perform any cleanup.
    """

    def process_duration_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        timestamp: Timestamp,
        wall_clock_time_ns: int,
        global_rank: int,
        pid: int,
    ) -> None:
        """Called by the :class:`~composer.profiler.profiler.Profiler` whenever there is a duration event to record.

        This method is called twice for each duration event -- once with ``is_start = True``,
        and then again with ``is_start = False``. Interleaving events are not permitted.
        Specifically, for each event (identified by the ``name``), a call with ``is_start = True`` will be followed
        by a call with ``is_start = False`` before another call with ``is_start = True``.

        Args:
            name (str): The name of the event.
            categories (Union[List[str], Tuple[str, ...]]): The categories for the event.
            is_start (bool): Whether the event is a start event or end event.
            timestamp (Timestamp): Snapshot of the training time.
            wall_clock_time_ns (int): The :py:func:`time.time_ns` corresponding to the event.
            global_rank (int): The ``global_rank`` corresponding to the event.
            pid (int): The ``pid`` corresponding to the event.
        """
        del name, categories, is_start, timestamp, wall_clock_time_ns, global_rank, pid  # unused
        pass

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        timestamp: Timestamp,
        wall_clock_time_ns: int,
        global_rank: int,
        pid: int,
    ) -> None:
        """Called by the :class:`~composer.profiler.profiler.Profiler` whenever there is an instant event to record.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            timestamp (Timestamp): Snapshot of current training time.
            wall_clock_time_ns (int): The :py:func:`time.time_ns` corresponding to the event.
            global_rank (int): The ``global_rank`` corresponding to the event.
            pid (int): The ``pid`` corresponding to the event.
        """
        del name, categories, timestamp, wall_clock_time_ns, global_rank, pid  # unused
        pass

    def process_counter_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        wall_clock_time_ns: int,
        global_rank: int,
        pid: int,
        values: Dict[str, Union[int, float]],
    ) -> None:
        """Called by the :class:`~composer.profiler.profiler.Profiler` whenever there is an counter event to record.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            wall_clock_time_ns (int): The :py:func:`time.time_ns` corresponding to the event.
            global_rank (int): The ``global_rank`` corresponding to the event.
            pid (int): The ``pid`` corresponding to the event.
            values (Dict[str, int | float]): The values corresponding to this counter event.
        """
        del name, categories, wall_clock_time_ns, global_rank, pid, values  # unused
        pass
