# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base classes, functions, and variables for logger."""

from __future__ import annotations

import collections.abc
import operator
import pathlib
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import torch

from composer.utils import ensure_tuple
from composer.utils.file_helpers import format_name_with_dist

if TYPE_CHECKING:
    from composer.core.state import State
    from composer.loggers.logger_destination import LoggerDestination

__all__ = ['LoggerDestination', 'Logger', 'format_log_data_value']


class Logger:
    """An interface to record training data.

    The :class:`.Trainer`, instances of :class:`.Callback`, and
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

    def log_traces(self, traces: Dict[str, Any]):
        for destination in self.destinations:
            destination.log_traces(traces)

    def log_hyperparameters(self, parameters: Dict[str, Any]):
        for destination in self.destinations:
            destination.log_hyperparameters(parameters)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not step:
            step = self._state.timestamp.batch.value
        for destination in self.destinations:
            destination.log_metrics(metrics, step)

    def data_fit(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError(
            'data_fit is no longer a valid call to the logger API. Please use log_hyperparameters or log_metrics instead'
        )

    def data_epoch(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError(
            'data_epoch is no longer a valid call to the logger API. Please use log_hyperparameters or log_metrics instead'
        )

    def data_batch(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError(
            'data_batch is no longer a valid call to the logger API. Please use log_hyperparameters or log_metrics instead'
        )

    def upload_file(
        self,
        remote_file_name: str,
        file_path: Union[pathlib.Path, str],
        *,
        overwrite: bool = False,
    ):
        """Upload ``file_path`` as a file named ``remote_file_name``.

        Both ``file_path`` and ``remote_file_name`` can be specified as format strings.
        See :func:`~.composer.utils.file_helpers.format_name_with_dist` for more information.

        .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

        Args:
            remote_file_name (str): A format string for the name of the file.
            file_path (str | pathlib.Path): A format string for the file path.
            overwrite (bool, optional): Whether to overwrite an existing file with the same ``remote_file_name``.
                (default: ``False``)
        """
        file_path = format_name_with_dist(format_str=str(file_path), run_name=self._state.run_name)
        file_path = pathlib.Path(file_path)
        for destination in self.destinations:
            destination.upload_file(
                state=self._state,
                remote_file_name=format_name_with_dist(format_str=remote_file_name, run_name=self._state.run_name),
                file_path=file_path,
                overwrite=overwrite,
            )

    def has_file_upload_destination(self) -> bool:
        """Determines if the logger has a destination which supports uploading files.

            Needed for checking if a model can be exported via this logger.

        Returns:
            bool: Whether any of the destinations support uploading files.
        """
        for destination in self.destinations:
            if destination.can_upload_files():
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
