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
import pathlib
import time
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import coolname
import torch

from composer.utils import dist, ensure_tuple

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
        run_name (str, optional): The name for this training run.

            If not specified, the timestamp will be combined with a :doc:`coolname <coolname:index>` like the
            following:

            .. testsetup:: composer.loggers.logger.Logger.__init__.run_name

                import random
                import coolname
                import time

                coolname.replace_random(random.Random(0))

                original_time = time.time

                time.time = lambda: 1647293526.1849217

            .. doctest:: composer.loggers.logger.Logger.__init__.run_name

                >>> logger = Logger(state=state, destinations=[])
                >>> logger.run_name
                '1647293526-electric-zebra'

            .. testcleanup:: composer.loggers.logger.Logger.__init__.run_name

                time.time = original_time

    Attributes:
        destinations (Sequence[LoggerDestination]):
            A sequence of :class:`~.LoggerDestination` to where logging calls will be sent.
        run_name (str): The ``run_name``.
    """

    def __init__(
        self,
        state: State,
        destinations: Optional[Union[LoggerDestination, Sequence[LoggerDestination]]] = None,
        run_name: Optional[str] = None,
    ):
        self.destinations = ensure_tuple(destinations)
        if run_name is None:
            # prefixing with the time so experiments sorted alphabetically will
            # have the latest experiment last
            run_name = str(int(time.time())) + "-" + coolname.generate_slug(2)
            run_name_list = [run_name]
            # ensure all ranks have the same experiment name
            dist.broadcast_object_list(run_name_list)
            run_name = run_name_list[0]
        self.run_name = run_name
        self._state = state

    def data(self, log_level: Union[str, int, LogLevel], data: LoggerDataDict) -> None:
        """Log data to the :attr:`destinations`.

        Args:
            log_level (str | int | LogLevel): The log level, which can be a name, value, or instance of
                :class:`LogLevel`.
            data (LoggerDataDict): The data to log.
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

        Args:
            log_level (str | int | LogLevel): The log level, which can be a name, value, or instance of
                :class:`LogLevel`.
            artifact_name (str): The name of the artifact.
            file_path (str | pathlib.Path): The file path.
            overwrite (bool, optional): Whether to overwrite an existing artifact with the same ``artifact_name``.
                (default: ``False``)
        """
        log_level = LogLevel(log_level)
        file_path = pathlib.Path(file_path)
        for destination in self.destinations:
            destination.log_file_artifact(
                state=self._state,
                log_level=log_level,
                artifact_name=artifact_name,
                file_path=file_path,
                overwrite=overwrite,
            )

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
