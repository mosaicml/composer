# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for logger callback."""

from __future__ import annotations

import pathlib
from abc import ABC
from typing import Any, Dict

from composer.core.callback import Callback
from composer.core.state import State
from composer.loggers.logger import LogLevel

__all__ = ['LoggerDestination']


class LoggerDestination(Callback, ABC):
    """Base class for logger destination.

    As this class extends :class:`~.callback.Callback`, logger destinations can run on any training loop
    :class:`.Event`. For example, it may be helpful to run on
    :attr:`.Event.EPOCH_END` to perform any flushing at the end of every epoch.

    Example:
        .. doctest::

            >>> from composer.loggers import LoggerDestination
            >>> class MyLogger(LoggerDestination):
            ...     def log_data(self, state, log_level, data):
            ...         print(f'Batch {int(state.timestamp.batch)}: {data}')
            >>> logger = MyLogger()
            >>> trainer = Trainer(
            ...     ...,
            ...     loggers=[logger]
            ... )
            Batch 0: {'rank_zero_seed': ...}
    """

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        """Log data.

        Subclasses should implement this method to store logged data (e.g. write it to a file, send it to a server,
        etc...). However, not all loggers need to implement this method.

        .. note::

            This method will block the training loop. For optimal performance, it is recommended to deepcopy the
            ``data`` (e.g. ``copy.deepcopy(data)``), and store the copied data in queue. Then, either:

            *   Use background thread(s) or process(s) to read from this queue to perform any I/O.
            *   Batch the data together and flush periodically on events, such as
                :attr:`.Event.BATCH_END` or :attr:`.Event.EPOCH_END`.

                .. seealso:: :class:`~composer.loggers.file_logger.FileLogger` as an example.

        Args:
            state (State): The training state.
            log_level (LogLevel): The log level.
            data (Dict[str, Any]): The data to log.
        """
        del state, log_level, data  # unused
        pass

    def log_file_artifact(
        self,
        state: State,
        log_level: LogLevel,
        artifact_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        """Handle logging of a file artifact stored at ``file_path`` to an artifact named ``artifact_name``.

        Subclasses should implement this method to store logged files (e.g. copy it to another folder or upload it to
        an object store), then it should implement this method. However, not all loggers need to implement this method.
        For example, the :class:`.TQDMLogger` does not implement this method, as it cannot
        handle file artifacts.

        .. note::

            *   This method will block the training loop. For optimal performance, it is recommended that this
                method copy the file to a temporary directory, enqueue the copied file for processing, and return.
                Then, use a background thread(s) or process(s) to read from this queue to perform any I/O.
            *   After this method returns, training can resume, and the contents of ``file_path`` may change (or be may
                deleted). Thus, if processing the file in the background (as is recommended), it is necessary to first
                copy the file to a temporary directory. Otherwise, the original file may no longer exist, or the logged
                artifact can be corrupted (e.g., if the logger destination is reading from file while the training loop
                is writing to it).

        .. seealso:: :doc:`Artifact Logging</trainer/artifact_logging>` for notes for file artifact logging.

        Args:
            state (State): The training state.
            log_level (Union[str, LogLevel]): A :class:`LogLevel`.
            artifact_name (str): The name of the artifact.
            file_path (pathlib.Path): The file path.
            overwrite (bool, optional): Whether to overwrite an existing artifact with the same ``artifact_name``.
                (default: ``False``)
        """
        del state, log_level, artifact_name, file_path, overwrite  # unused
        pass

    def get_file_artifact(
        self,
        artifact_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        """Handle downloading an artifact named ``artifact_name`` to ``destination``.

        Args:
            artifact_name (str): The name of the artifact.
            destination (str): The destination filepath.
            overwrite (bool): Whether to overwrite an existing file at ``destination``. Defaults to ``False``.
            progress_bar (bool, optional): Whether to show a progress bar. Ignored if ``path`` is a local file.
                (default: ``True``)
        """
        del artifact_name, destination, overwrite, progress_bar  # unused
        raise NotImplementedError

    def can_log_file_artifacts(self) -> bool:
        """Indicates whether LoggerDestination can log file artifacts.

        Defaults to false, should return True for derived logger classes that implement log_file_artifact().

        Returns:
            bool: Whether the class supports logging file artifacts.
        """
        return False
