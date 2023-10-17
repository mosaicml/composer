# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base classes, functions, and variables for logger."""

from __future__ import annotations

import collections.abc
import operator
import pathlib
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from composer.utils import ensure_tuple, format_name_with_dist

if TYPE_CHECKING:
    from composer.core import State
    from composer.loggers import (CometMLLogger, ConsoleLogger, FileLogger, InMemoryLogger, MLFlowLogger,
                                  MosaicMLLogger, ProgressBarLogger, RemoteUploaderDownloader, SlackLogger,
                                  TensorboardLogger, WandBLogger)
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

    def log_table(self, columns: List[str], rows: List[List[Any]], name: str = 'Table') -> None:
        for destination in self.destinations:
            destination.log_table(columns, rows, name)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is None:
            step = self._state.timestamp.batch.value
        for destination in self.destinations:
            destination.log_metrics(metrics, step)

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'Images',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[Dict[int, str]] = None,
        use_table: bool = True,
    ):
        """Log images. Logs any tensors or arrays as images.

        Args:
            images (np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor]): Dictionary mapping
                image(s)' names (str) to an image of array of images.
            name (str): The name of the image(s). (Default: ``'Images'``)
            channels_last (bool): Whether the channel dimension is first or last.
                (Default: ``False``)
            step (int, optional): The current step or batch of training at the
                time of logging. Defaults to None. If not specified the specific
                LoggerDestination implementation will choose a step (usually a running
                counter).
            masks (Dict[str, np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor]], optional): A dictionary
                mapping the mask name (e.g. predictions or ground truth) to a sequence of masks.
            mask_class_labels (Dict[int, str], optional): Dictionary mapping label id to its name. Used for labelling
                each color in the mask.
            use_table (bool): Whether to make a table of the images or not. (default: ``True``). Only for use
                with WandB.
        """
        if step is None:
            step = self._state.timestamp.batch.value
        for destination in self.destinations:
            destination.log_images(images, name, channels_last, step, masks, mask_class_labels, use_table)

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

    @property
    def wandb(self) -> Optional[Sequence['WandBLogger']]:
        """Returns instances of the :class:`~.WandBLogger` destination, if it/they exist.

        Returns:
            Optional[Sequence[WandBLogger]]: The WandBLogger instance(s), if it/they exist.

        """
        from composer.loggers.wandb_logger import WandBLogger
        return [destination for destination in self.destinations if isinstance(destination, WandBLogger)]

    @property
    def mlflow(self) -> Optional[Sequence['MLFlowLogger']]:
        """Returns instances of the :class:`~.MLFlowLogger` destination, if it/they exist.

        Returns:
            Optional[Sequence[MLFlowLogger]]: The MLFlowLogger instance(s), if it/they exist.
        """
        from composer.loggers.mlflow_logger import MLFlowLogger
        return [destination for destination in self.destinations if isinstance(destination, MLFlowLogger)]

    @property
    def remote_uploader_downloader(self) -> Optional[Sequence['RemoteUploaderDownloader']]:
        """Returns instances of the :class:`~.RemoteUploaderDownloader` destination, if it/they exist.

        Returns:
            Optional[Union[RemoteUploaderDownloader, Sequence[RemoteUploaderDownloader]]]: The RemoteUploaderDownloader instance(s), if it/they exist.
        """
        from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
        return [destination for destination in self.destinations if isinstance(destination, RemoteUploaderDownloader)]

    @property
    def mosaicml(self) -> Optional[Sequence['MosaicMLLogger']]:
        """Returns instances of the :class:`~.MosaicMLLogger` destination, if it/they exist.

        Returns:
            Optional[Union[MosaicMLLogger, Sequence[MosaicMLLogger]]]: The MosaicMLLogger instance(s), if it/they exist.
        """
        from composer.loggers.mosaicml_logger import MosaicMLLogger
        return [destination for destination in self.destinations if isinstance(destination, MosaicMLLogger)]

    @property
    def slack(self) -> Optional[Sequence['SlackLogger']]:
        """Returns instances of the :class:`~.SlackLogger` destination, if it/they exist.

        Returns:
            Optional[Union[SlackLogger, Sequence[SlackLogger]]]: The SlackLogger instance(s), if it/they exist.
        """
        from composer.loggers.slack_logger import SlackLogger
        return [destination for destination in self.destinations if isinstance(destination, SlackLogger)]

    @property
    def console(self) -> Optional[Sequence['ConsoleLogger']]:
        """Returns instances of the :class:`~.ConsoleLogger` destination, if it/they exist.

        Returns:
            Optional[Union[ConsoleLogger, Sequence[ConsoleLogger]]]: The ConsoleLogger instance(s), if it/they exist.
        """
        from composer.loggers.console_logger import ConsoleLogger
        return [destination for destination in self.destinations if isinstance(destination, ConsoleLogger)]

    @property
    def file(self) -> Optional[Sequence['FileLogger']]:
        """Returns instances of the :class:`~.FileLogger` destination, if it/they exist.

        Returns:
            Optional[Union[FileLogger, Sequence[FileLogger]]]: The FileLogger instance(s), if it/they exist.
        """
        from composer.loggers.file_logger import FileLogger
        return [destination for destination in self.destinations if isinstance(destination, FileLogger)]

    @property
    def in_memory(self) -> Optional[Sequence['InMemoryLogger']]:
        """Returns instances of the :class:`~.InMemoryLogger` destination, if it/they exist.

        Returns:
            Optional[Union[InMemoryLogger, Sequence[InMemoryLogger]]]: The InMemoryLogger instance(s), if it/they exist.
        """
        from composer.loggers.in_memory_logger import InMemoryLogger
        return [destination for destination in self.destinations if isinstance(destination, InMemoryLogger)]

    @property
    def progress_bar(self) -> Optional[Sequence['ProgressBarLogger']]:
        """Returns instances of the :class:`~.ProgressBarLogger` destination, if it/they exist.

        Returns:
            Optional[Union[ProgressBarLogger, Sequence[ProgressBarLogger]]]: The ProgressBarLogger instance(s), if it/they exist.
        """
        from composer.loggers.progress_bar_logger import ProgressBarLogger
        return [destination for destination in self.destinations if isinstance(destination, ProgressBarLogger)]

    @property
    def tensorboard(self) -> Optional[Sequence['TensorboardLogger']]:
        """Returns instances of the :class:`~.TensorboardLogger` destination, if it/they exist.

        Returns:
            Optional[Union[TensorboardLogger, Sequence[TensorboardLogger]]]: The TensorboardLogger instance(s), if it/they exist.
        """
        from composer.loggers.tensorboard_logger import TensorboardLogger
        return [destination for destination in self.destinations if isinstance(destination, TensorboardLogger)]

    @property
    def cometml(self) -> Optional[Sequence['CometMLLogger']]:
        """Returns instances of the :class:`~.CometMLLogger` destination, if it/they exist.

        Returns:
            Optional[Union[CometMLLogger, Sequence[CometMLLogger]]]: The CometMLLogger instance(s), if it/they exist.
        """
        from composer.loggers.cometml_logger import CometMLLogger
        return [destination for destination in self.destinations if isinstance(destination, CometMLLogger)]


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
