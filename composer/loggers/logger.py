# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base classes, functions, and variables for logger."""

from __future__ import annotations

import collections.abc
import operator
import pathlib
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import torch

from composer.utils import ensure_tuple
from composer.utils.file_helpers import format_name_with_dist

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.loggers.logger_destination import LoggerDestination

__all__ = ['LoggerDestination', 'Logger', 'LogLevel', 'format_log_data_value']


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

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, LogLevel):
            return value
        if isinstance(value, int):
            return LogLevel(value)
        if isinstance(value, str):
            return LogLevel[value.upper()]
        return super()._missing_(value)


class Logger:
    """An interface to record training data.

    The :class:`~composer.trainer.trainer.Trainer`, instances of :class:`~composer.core.callback.Callback`, and
    instances of :class:`~composer.core.algorithm.Algorithm` invoke the logger to record data such as
    the epoch, training loss, and custom metrics as provided by individual callbacks and algorithms.
    This class does not store any data itself; instead, it routes all data to the ``destinations``.
    Each destination (e.g. the :class:`~composer.loggers.file_logger.FileLogger`,
    :class:`~composer.loggers.in_memory_logger.InMemoryLogger`) is responsible for storing the data itself
    (e.g. writing it to a file or storing it in memory).

    Args:
        state (State): The training state.
        destinations (LoggerDestination | Sequence[LoggerDestination], optional):
            The logger destinations, to where logging data will be sent. (default: ``None``)

    Attributes:
        destinations (Sequence[LoggerDestination]):
            A sequence of :class:`~.LoggerDestination` to where logging calls will be sent.
    """

    def __init__(
        self,
        state: State,
        destinations: Optional[Union[LoggerDestination, Sequence[LoggerDestination]]] = None,
    ):
        self.destinations = ensure_tuple(destinations)
        self._state = state

    def data(self, log_level: Union[str, int, LogLevel], data: Dict[str, Any]) -> None:
        """Log data to the :attr:`destinations`.

        Args:
            log_level (str | int | LogLevel): The log level, which can be a name, value, or instance of
                :class:`LogLevel`.
            data (Dict[str, Any]): The data to log.
        """
        log_level = LogLevel(log_level)

        for destination in self.destinations:
            destination.log_data(self._state, log_level, data)

    def file_artifact(
        self,
        log_level: Union[str, int, LogLevel],
        artifact_name: str,
        file_path: Union[pathlib.Path, str],
        *,
        overwrite: bool = False,
    ):
        """Log ``file_path`` as an artifact named ``artifact_name``.

        Both ``file_path`` and ``artifact_name`` can be specified as format strings.
        See :func:`~.composer.utils.file_helpers.format_name_with_dist` for more information.

        Args:
            log_level (str | int | LogLevel): The log level, which can be a name, value, or instance of
                :class:`LogLevel`.
            artifact_name (str): A format string for the name of the artifact.
            file_path (str | pathlib.Path): A format string for the file path.
            overwrite (bool, optional): Whether to overwrite an existing artifact with the same ``artifact_name``.
                (default: ``False``)
        """
        log_level = LogLevel(log_level)
        file_path = format_name_with_dist(format_str=str(file_path), run_name=self._state.run_name)
        file_path = pathlib.Path(file_path)
        for destination in self.destinations:
            destination.log_file_artifact(
                state=self._state,
                log_level=log_level,
                artifact_name=format_name_with_dist(format_str=artifact_name, run_name=self._state.run_name),
                file_path=file_path,
                overwrite=overwrite,
            )

    def data_fit(self, data: Dict[str, Any]) -> None:
        """Helper function for ``self.data(LogLevel.FIT, data)``."""
        self.data(LogLevel.FIT, data)

    def data_epoch(self, data: Dict[str, Any]) -> None:
        """Helper function for ``self.data(LogLevel.EPOCH, data)``."""
        self.data(LogLevel.EPOCH, data)

    def data_batch(self, data: Dict[str, Any]) -> None:
        """Helper function for ``self.data(LogLevel.BATCH, data)``."""
        self.data(LogLevel.BATCH, data)

    def has_file_artifact_destination(self) -> bool:
        """Determines if the logger has a destination which supports logging file artifacts.

            Needed for checking if a model can be exported via this logger.

        Returns:
            bool: Whether any of the destinations has supports file artifacts.
        """
        for destination in self.destinations:
            if destination.can_log_file_artifacts():
                return True
        return False


def format_log_data_value(data: Any) -> str:
    """Recursively formats a given log data value into a string.

    Args:
        data: Data to format.

    Returns:
        str: ``data`` as a string.
    """
    if data is None:
        return 'None'
    if isinstance(data, str):
        return f"\"{data}\""
    if isinstance(data, int):
        return str(data)
    if isinstance(data, float):
        return f'{data:.4f}'
    if isinstance(data, torch.Tensor):
        if data.shape == () or reduce(operator.mul, data.shape, 1) == 1:
            return format_log_data_value(data.cpu().item())
        return 'Tensor of shape ' + str(data.shape)
    if isinstance(data, collections.abc.Mapping):
        output = ['{ ']
        for k, v in data.items():
            assert isinstance(k, str)
            v = format_log_data_value(v)
            output.append(f"\"{k}\": {v}, ")
        output.append('}')
        return ''.join(output)
    if isinstance(data, collections.abc.Iterable):
        return '[' + ', '.join(format_log_data_value(v) for v in data) + ']'

    # Unknown format catch-all
    return str(data)
