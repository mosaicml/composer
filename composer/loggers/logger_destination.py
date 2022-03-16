# Copyright 2021 MosaicML. All Rights Reserved.

"""Base class for logger callback."""

from __future__ import annotations

import pathlib
from abc import ABC

from composer.core.callback import Callback
from composer.core.state import State
from composer.loggers.logger import LoggerDataDict, LogLevel

__all__ = ["LoggerDestination"]


class LoggerDestination(Callback, ABC):
    """Base class for logger destination.

    As this class extends :class:`~.callback.Callback`, logger destinations can run on any training loop
    :class:`~composer.core.event.Event`. For example, it may be helpful to run on
    :attr:`~composer.core.event.Event.EPOCH_END` to perform any flushing at the end of every epoch.

    Example
    -------

    >>> from composer.loggers import LoggerDestination
    >>> class MyLogger(LoggerDestination):
    ...     def log_data(self, state, log_level, data):
    ...         print(f'Batch {int(state.timer.batch)}: {log_level} {data}')
    >>> logger = MyLogger()
    >>> trainer = Trainer(
    ...     ...,
    ...     loggers=[logger]
    ... )
    """

    def log_data(self, state: State, log_level: LogLevel, data: LoggerDataDict):
        """Log data.

        Subclasses should implement this method to store logged data (e.g. write it to a file, send it to a server,
        etc...). However, not all loggers need to implement this method.

        .. note::

            This method will block the training loop. For optimal performance, it is recommended to deepcopy the
            ``data`` (e.g. ``copy.deepcopy(data)``), and store the copied data in queue. Then, either:

            *   Use background thread(s) or process(s) to read from this queue to perform any I/O.
            *   Batch the data together and flush periodically on events, such as
                :attr:`~composer.core.event.Event.BATCH_END` or :attr:`~composer.core.event.Event.EPOCH_END`.

                .. seealso:: :class:`~composer.loggers.file_logger.FileLogger` as an example.

        Args:
            state (State): The training state.
            log_level (LogLevel): The log level.
            data (LoggerDataDict): The data to log.
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
        For example, the :class:`~composer.loggers.tqdm_logger.TQDMLogger` does not implement this method, as it cannot
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

        Args:
            state (State): The training state.
            log_level (Union[str, LogLevel]): A :class:`LogLevel`.
            artifact_name (str): The name of the artifact.
            file_path (str | pathlib.Path): The file path.
            overwrite (bool, optional): Whether to overwrite an existing artifact with the same ``artifact_name``.
                (default: ``False``)
        """
        del state, log_level, artifact_name, file_path, overwrite  # unused
        pass
