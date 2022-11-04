# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Profiler Trace Handler."""

from __future__ import annotations

import abc
import pathlib
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from composer.core.callback import Callback

if TYPE_CHECKING:
    from composer.core import Timestamp

__all__ = ['TraceHandler']


class TraceHandler(Callback, abc.ABC):
    """Base class for Composer Profiler trace handlers.

    Subclasses should implement :meth:`process_duration_event`, :meth:`process_instant_event`,
    :meth:`process_counter_event`, and :meth:`process_chrome_json_trace_file` to record trace events.

    Since :class:`TraceHandler` subclasses :class:`.Callback`, a trace handler can run on any
    :class:`.Event` (such as on :attr:`.Event.INIT` to open files or on :attr:`.Event.BATCH_END` to periodically dump
    data to files) and use :meth:`.Callback.close` to perform any cleanup.
    """

    def process_duration_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        timestamp: Timestamp,
        wall_clock_time_ns: int,
    ) -> None:
        """Invoked whenever there is a duration event to record.

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
        """
        del name, categories, is_start, timestamp, wall_clock_time_ns  # unused
        pass

    def process_instant_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        timestamp: Timestamp,
        wall_clock_time_ns: int,
    ) -> None:
        """Invoked whenever there is an instant event to record.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            timestamp (Timestamp): Snapshot of current training time.
            wall_clock_time_ns (int): The :py:func:`time.time_ns` corresponding to the event.
        """
        del name, categories, timestamp, wall_clock_time_ns  # unused
        pass

    def process_counter_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        timestamp: Timestamp,
        wall_clock_time_ns: int,
        values: Dict[str, Union[int, float]],
    ) -> None:
        """Invoked whenever there is an counter event to record.

        Args:
            name (str): The name of the event.
            categories (List[str] | Tuple[str, ...]): The categories for the event.
            timestamp (Timestamp): The timestamp.
            wall_clock_time_ns (int): The :py:func:`time.time_ns` corresponding to the event.
            values (Dict[str, int | float]): The values corresponding to this counter event.
        """
        del name, categories, timestamp, wall_clock_time_ns, values  # unused
        pass

    def process_chrome_json_trace_file(self, filepath: pathlib.Path) -> None:
        """Invoked when there are events in Chrome JSON format to record.

        See `this document <https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview>`_
        for more information.

        Args:
            filepath (pathlib.Path): The filepath to a Chrome JSON trace file.
        """
        del filepath  # unused
        pass
