# Copyright 2021 MosaicML. All Rights Reserved.

"""Base classes, functions, and variables for logger.

Attributes:

     LoggerData: Data value(s) to be logged. Can be any of the following types:
         ``str``; ``float``; ``int``; :class:`torch.Tensor`; ``Sequence[LoggerData]``;
         ``Mapping[str, LoggerData]``.
     LoggerDataDict: Name-value pair for data to be logged. Type ``Mapping[str, LoggerData]``.
         Example: ``{"accuracy", 21.3}``.
"""

from __future__ import annotations

import collections.abc
import operator
from copy import deepcopy
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Sequence, Union

import torch

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.loggers.logger_destination import LoggerDestination

__all__ = ["LoggerDestination", "Logger", "LogLevel", "LoggerData", "LoggerDataDict", "format_log_data_value"]

LoggerData = Union[str, float, int, torch.Tensor, List["LoggerData"], Dict[str, "LoggerData"]]
LoggerDataDict = Dict[str, LoggerData]


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
    """An interface to record training data.

    The :class:`~composer.trainer.trainer.Trainer`, :class:`~composer.core.callback.Callback`\\s, and
    :class:`~composer.core.algorithm.Algorithm`\\s invoke the logger to record data such as
    the epoch, training loss, and custom metrics as provided by individual callbacks and algorithms.

    This class does not store any data itself; instead, it routes all data to the ``logger_destinations``.
    Each destination (e.g. the :class:`~composer.loggers.file_logger.FileLogger`,
    :class:`~composer.loggers.in_memory_logger.InMemoryLogger`) is responsible for storing the data itself
    (e.g. writing it to a file or storing it in memory).

    Args:
        state (State): The training state.
        destinations (Sequence[LoggerDestination]):
            The logger destinations, to where logging data will be sent.

    Attributes:
        destinations (Sequence[LoggerDestination]):
            A sequence of :class:`~.LoggerDestination`\\s to which logging calls will be sent.
    """

    def __init__(
            self,
            state: State,
            destinations: Sequence[LoggerDestination] = tuple(),
    ):
        self.destinations = destinations
        self._state = state

    def _get_destinations_for_log_level(self, log_level: LogLevel) -> Generator[LoggerDestination, None, None]:
        for destination in self.destinations:
            if destination.will_log(self._state, log_level):
                yield destination

    def data(self, log_level: Union[str, LogLevel], data: Union[LoggerDataDict, Callable[[], LoggerDataDict]]) -> None:
        """Log a metric to the :attr:`destinations`.

        Args:
            log_level (Union[str, LogLevel]): A :class:`LogLevel`.
            data (Union[LoggerDataDict, Callable[[], LoggerDataDict]]):
                Can be either logging data or a callable that returns data to be logged.
                Callables will be invoked only when
                :meth:`~composer.loggers.logger_destination.LoggerDestination.will_log` returns True for at least one
                :class:`~.composer.loggers.logger_destination.LoggerDestination`. Useful when it is
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
            destination.log_data(self._state.timer.get_timestamp(), log_level, copied_data)

    def data_fit(self, data: LoggerDataDict) -> None:
        """Helper function for ``self.data(LogLevel.FIT, data)``"""
        self.data(LogLevel.FIT, data)

    def data_epoch(self, data: LoggerDataDict) -> None:
        """Helper function for ``self.data(LogLevel.EPOCH, data)``"""
        self.data(LogLevel.EPOCH, data)

    def data_batch(self, data: LoggerDataDict) -> None:
        """Helper function for ``self.data(LogLevel.BATCH, data)``"""
        self.data(LogLevel.BATCH, data)


def format_log_data_value(data: LoggerData) -> str:
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
