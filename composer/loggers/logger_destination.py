# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for logger callback."""

from __future__ import annotations

import pathlib
from abc import ABC
from typing import Any, Dict, Optional

from composer.core.callback import Callback
from composer.core.state import State

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
            Batch 0: {'num_nodes': ...}
            Batch 0: {'rank_zero_seed': ...}
    """

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters, configurations, and settings.

        Logs any parameter/configuration/setting that doesn't vary during the run.

        Args:
            hyperparameters (Dict[str, Any]): A dictionary mapping hyperparameter names
                (strings) to their values (Any).
        """
        del hyperparameters  # unused
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics or parameters that vary during training.

        Args:
            metrics (Dict[str, float]): Dictionary mapping metric name (str) to metric
                scalar value (float)
            step (Optional[int], optional): The current step or batch of training at the
                time of logging. Defaults to None. If not specified the specific
                LoggerDestination implementation will choose a step (usually a running
                counter).
        """
        del metrics, step  # unused
        pass

    def log_traces(self, traces: Dict[str, Any]):
        """Log traces. Logs any debug-related data like algorithm traces.

        Args:
            traces (Dict[str, float]): Dictionary mapping trace names (str) to trace
                (Any).
        """
        del traces
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
