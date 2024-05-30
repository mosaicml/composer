# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for logger callback."""

from __future__ import annotations

import pathlib
from abc import ABC
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import torch

from composer.core.callback import Callback

if TYPE_CHECKING:
    from composer.core import State

__all__ = ['LoggerDestination']


class LoggerDestination(Callback, ABC):
    """Base class for logger destination.

    As this class extends :class:`~.callback.Callback`, logger destinations can run on any training loop
    :class:`.Event`. For example, it may be helpful to run on
    :attr:`.Event.EPOCH_END` to perform any flushing at the end of every epoch.

    Example:
        .. doctest::

            >>> from composer.loggers import LoggerDestination
            >>> from composer.trainer import Trainer
            >>> class MyLogger(LoggerDestination):
            ...     def log_hyperparameters(self, data):
            ...         print(f'Batch {int(state.timestamp.batch)}: {data}')
            >>> logger = MyLogger()
            >>> trainer = Trainer(
            ...     ...,
            ...     loggers=[logger]
            ... )
            Batch 0: {'composer_version': ...}
            Batch 0: {'composer_commit_hash': ...}
            Batch 0: {'num_nodes': ...}
            Batch 0: {'rank_zero_seed': ...}
    """

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        """Log hyperparameters, configurations, and settings.

        Logs any parameter/configuration/setting that doesn't vary during the run.

        Args:
            hyperparameters (dict[str, Any]): A dictionary mapping hyperparameter names
                (strings) to their values (Any).
        """
        del hyperparameters  # unused
        pass

    def log_table(
        self,
        columns: list[str],
        rows: list[list[Any]],
        name: str = 'Table',
        step: Optional[int] = None,
    ) -> None:
        """Log a table.

        Args:
            columns (list[str]): Names of the columns in the table.
            rows (list[list[Any]]): 2D row-oriented array of values.
            name (str): Name of table. (Default: ``'Table'``)
            step (Optional[int], optional): The current step or batch of training at the
                time of logging. Defaults to None. If not specified the specific
                LoggerDestination implementation will choose a step (usually a running
                counter).
        """
        del columns, rows, name, step
        pass

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics or parameters that vary during training.

        Args:
            metrics (dict[str, float]): Dictionary mapping metric name (str) to metric
                scalar value (float)
            step (Optional[int], optional): The current step or batch of training at the
                time of logging. Defaults to None. If not specified the specific
                LoggerDestination implementation will choose a step (usually a running
                counter).
        """
        del metrics, step  # unused
        pass

    def log_traces(self, traces: dict[str, Any]):
        """Log traces. Logs any debug-related data like algorithm traces.

        Args:
            traces (dict[str, float]): Dictionary mapping trace names (str) to trace
                (Any).
        """
        del traces
        pass

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'Images',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[dict[int, str]] = None,
        use_table: bool = True,
    ):
        """Log images. Logs any tensors or arrays as images.

        Args:
            images (np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor]): Dictionary mapping
                image(s)' names (str) to an image of array of images.
            name (str): The name of the image(s). (Default: ``'Images'``)
            channels_last (bool): Whether the channel dimension is first or last.
                (Default: ``False``)
            step (Optional[int], optional): The current step or batch of training at the
                time of logging. Defaults to None. If not specified the specific
                LoggerDestination implementation will choose a step (usually a running
                counter).
            masks (dict[str, np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor]], optional): A dictionary
                mapping the mask name (e.g. predictions or ground truth) to a sequence of masks.
            mask_class_labels (dict[int, str], optional): Dictionary mapping label id to its name. Used for labelling
                each color in the mask.
            use_table (bool): Whether to make a table of the images or not. (default: ``True``). Only for use
                with WandB.
        """
        del images, name, channels_last, step, masks, mask_class_labels, use_table
        pass

    def upload_file(
        self,
        state: State,
        remote_file_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        """Handle uploading a file stored at ``file_path`` to a file named ``remote_file_name``.

        Subclasses should implement this method to store logged files (e.g. copy it to another folder or upload it to
        an object store). However, not all loggers need to implement this method.
        For example, the :class:`.TQDMLogger` does not implement this method, as it cannot
        handle file uploads.

        .. note::

            *   This method will block the training loop. For optimal performance, it is recommended that this
                method copy the file to a temporary directory, enqueue the copied file for processing, and return.
                Then, use a background thread(s) or process(s) to read from this queue to perform any I/O.
            *   After this method returns, training can resume, and the contents of ``file_path`` may change (or be may
                deleted). Thus, if processing the file in the background (as is recommended), it is necessary to first
                copy the file to a temporary directory. Otherwise, the original file may no longer exist, or the logged
                file can be corrupted (e.g., if the logger destination is reading from file while the training loop
                is writing to it).

        .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

        Args:
            state (State): The training state.
            remote_file_name (str): The name of the file.
            file_path (pathlib.Path): The file path.
            overwrite (bool, optional): Whether to overwrite an existing file with the same ``remote_file_name``.
                (default: ``False``)
        """
        del state, remote_file_name, file_path, overwrite  # unused
        pass

    def download_file(
        self,
        remote_file_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        """Handle downloading a file named ``remote_file_name`` to ``destination``.

        Args:
            remote_file_name (str): The name of the file.
            destination (str): The destination filepath.
            overwrite (bool): Whether to overwrite an existing file at ``destination``. Defaults to ``False``.
            progress_bar (bool, optional): Whether to show a progress bar. Ignored if ``path`` is a local file.
                (default: ``True``)
        """
        del remote_file_name, destination, overwrite, progress_bar  # unused
        raise NotImplementedError

    def can_upload_files(self) -> bool:
        """Indicates whether LoggerDestination can upload files.

        Defaults to false, should return True for derived logger classes that implement upload_file().

        Returns:
            bool: Whether the class supports uploading files.
        """
        return False
