# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import collections.abc
import operator
from copy import deepcopy
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Callable, Generator, Mapping, Sequence, Union

import torch

if TYPE_CHECKING:
    from composer.core.logging.base_backend import BaseLoggerBackend
    from composer.core.state import State

TLogDataValue = Union[str, float, int, torch.Tensor, Sequence["TLogDataValue"], Mapping[str, "TLogDataValue"]]
TLogData = Mapping[str, TLogDataValue]


class LogLevel(IntEnum):
    """LogLevel denotes where in the training loop log messages are generated.

    Logging destinations use the LogLevel to determine whether to record a given
    metric or state change.

    Attributes:
        FIT: Logged once per training run.
        EPOCH: Logged once per epoch.
        BATCH: Logged once per batch.
        MICROBATCH: Logged once per microbatch (e.g. forward pass).
        VERBOSE: Logged for debugging.
    """
    FIT = 1
    EPOCH = 2
    BATCH = 3
    MICROBATCH = 4
    VERBOSE = 5


class Logger:
    """Logger routes metrics to the
    :class:`~composer.core.logging.base_backend.BaseLoggerBackend`.

    Args:
        state (~composer.core.state.State):
            The global :class:`~composer.core.state.State` object.
        backends (Sequence[BaseLoggerBackend]):
            A sequence of
            :class:`~composer.core.logging.base_backend.BaseLoggerBackend`\s
            to which logging calls will be sent.

    Attributes:
        backends (Sequence[BaseLoggerBackend]):
            A sequence of
            :class:`~composer.core.logging.base_backend.BaseLoggerBackend`\s
            to which logging calls will be sent.
    """

    def __init__(
            self,
            state: State,
            backends: Sequence[BaseLoggerBackend] = tuple(),
    ):
        self.backends = backends
        self._state = state

    def _get_destinations_for_log_level(self, log_level: LogLevel) -> Generator[BaseLoggerBackend, None, None]:
        for destination in self.backends:
            if destination.will_log(self._state, log_level):
                yield destination

    def metric(self, log_level: Union[str, LogLevel], data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Log a metric to the :attr:`backends`.

        Args:
            log_level (Union[str, LogLevel]): A :class:`LogLevel`.
            data (Union[TLogData, Callable[[], TLogData]]):
                Can be either logging data or a callable that returns
                data to be logged. Callables will be invoked
                only when :meth:`~composer.core.logging.logger.Logger.will_log`
                returns True for at least one
                :class:`~composer.core.logging.base_backend.BaseLoggerBackend`.
                Useful when it is expensive to generate the data to be logged.
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
            destination.log_metric(self._state.epoch, self._state.step, log_level, copied_data)

    def metric_fit(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``metric(LogLevel.FIT, data)``"""
        self.metric(LogLevel.FIT, data)

    def metric_epoch(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``self.metric(LogLevel.EPOCH, data)``"""
        self.metric(LogLevel.EPOCH, data)

    def metric_batch(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``self.metric(LogLevel.BATCH, data)``"""
        self.metric(LogLevel.BATCH, data)

    def metric_microbatch(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``self.metric(LogLevel.MICROBATCH, data)``"""
        self.metric(LogLevel.MICROBATCH, data)

    def metric_verbose(self, data: Union[TLogData, Callable[[], TLogData]]) -> None:
        """Helper function for ``self.metric(LogLevel.VERBOSE, data)``"""
        self.metric(LogLevel.VERBOSE, data)


def format_log_data_value(data: TLogDataValue, /) -> str:
    """Recursively formats a given log data value into a string.

    Args:
        data: Data to format.

    Returns:
        str: The data, as a string.
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
