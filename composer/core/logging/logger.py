# Copyright 2021 MosaicML. All Rights Reserved.

"""Base classes, functions, and variables for logger.

Attributes:

     TLogDataValue: Data value(s) to be logged. Can be any of the following types:
         ``str``; ``float``; ``int``; :class:`torch.Tensor`; ``Sequence[TLogDataValue]``;
         ``Mapping[str, TLogDataValue]``.
     TLogData: Name-value pair for data to be logged. Type ``Mapping[str, TLogDataValue]``.
         Example: ``{"accuracy", 21.3}``.
"""

from __future__ import annotations

import collections.abc
import operator
from copy import deepcopy
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Callable, Generator, Mapping, Sequence, Union

import torch

if TYPE_CHECKING:
    from composer.core.logging.base_backend import LoggerCallback
    from composer.core.state import State

__all__ = ["LoggerCallback", "Logger", "LogLevel", "TLogData", "TLogDataValue", "format_log_data_value"]

TLogDataValue = Union[str, float, int, torch.Tensor, Sequence["TLogDataValue"], Mapping[str, "TLogDataValue"]]
TLogData = Mapping[str, TLogDataValue]


class LogLevel(IntEnum):
    """LogLevel denotes when in the training loop log messages are generated.

    Logging destinations use the LogLevel to determine whether to record a given
    metric or state change.

    Attributes:
        FIT: Logged once per training run.
        EPOCH: Logged once per epoch.
        BATCH: Logged once per batch.
    """
    FIT = 1
    EPOCH = 2
    BATCH = 3


class Logger:
    """Logger routes metrics to the :class:`.LoggerCallback`. Logger is what users call from within
    algorithms/callbacks. A logger routes the calls/data to any different number of destination
    :class:`.LoggerCallback`\\s (e.g., :class:`.FileLogger`, :class:`.InMemoryLogger`, etc.). Data to be logged should be
    of the type :attr:`~.logger.TLogData` (i.e., a ``{<name>: <value>}`` mapping).

    Args:
        state (State): The global :class:`~.core.state.State` object.
        backends (Sequence[LoggerCallback]): A sequence of :class:`.LoggerCallback`\\s to which logging calls will be sent.

    Attributes:
        backends (Sequence[LoggerCallback]):
            A sequence of :class:`~..base_backend.LoggerCallback`\\s to which logging calls will be sent.
    """

    def __init__(
            self,
            state: State,
            backends: Sequence[LoggerCallback] = tuple(),
    ):
        self.backends = backends
        self._state = state

    def _get_destinations_for_log_level(self, log_level: LogLevel) -> Generator[LoggerCallback, None, None]:
        for destination in self.backends:
            if destination.will_log(self._state, log_level):
                yield destination

    def metric(self, log_level: Union[str, LogLevel], data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Log a metric to the :attr:`backends`.

        Args:
            log_level (Union[str, LogLevel]): A :class:`LogLevel`.
            data (Union[TLogData, Callable[[], TLogData]]):
                Can be either logging data or a callable that returns data to be logged.
                Callables will be invoked only when
                :meth:`~.base_backend.LoggerCallback.will_log` returns True for at least one
                :class:`~.logging.base_backend.LoggerCallback`. Useful when it is
                expensive to generate the data to be logged.
        """
        if isinstance(log_level, str):
            log_level = LogLevel[log_level.upper()]

        for destination in self._get_destinations_for_log_level(log_level):
            if callable(data):
                data = data()
            # copying the data in case if a backend queues the logged data and flushes later
            # this way, the flushed data will be the same as at the time of the logger call
            copied_data = deepcopy(data)
            assert isinstance(copied_data, collections.abc.Mapping)
            destination.log_metric(self._state.timer.get_timestamp(), log_level, copied_data)

    def metric_fit(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``metric(LogLevel.FIT, data)``"""
        self.metric(LogLevel.FIT, data)

    def metric_epoch(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``self.metric(LogLevel.EPOCH, data)``"""
        self.metric(LogLevel.EPOCH, data)

    def metric_batch(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``self.metric(LogLevel.BATCH, data)``"""
        self.metric(LogLevel.BATCH, data)


def format_log_data_value(data: TLogDataValue) -> str:
    """Recursively formats a given log data value into a string.

    Args:
        data: Data to format.

    Returns:
        str: ``data`` as a string.
    """
    if data is None:
        return "None"
    if isinstance(data, str):
        return f"\"{data}\""
    if isinstance(data, int):
        return str(data)
    if isinstance(data, float):
        return f"{data:.4f}"
    if isinstance(data, torch.Tensor):
        if data.shape == tuple() or reduce(operator.mul, data.shape, 1) == 1:
            return format_log_data_value(data.cpu().item())
        return "Tensor of shape " + str(data.shape)
    if isinstance(data, collections.abc.Mapping):
        output = ['{ ']
        for k, v in data.items():
            assert isinstance(k, str)
            v = format_log_data_value(v)
            output.append(f"\"{k}\": {v}, ")
        output.append('}')
        return "".join(output)
    if isinstance(data, collections.abc.Iterable):
        return "[" + ", ".join(format_log_data_value(v) for v in data) + "]"
    raise NotImplementedError(f"Unable to format variable of type: {type(data)} with value {data}")
