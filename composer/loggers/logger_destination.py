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

    Subclasses must implement :meth:`log_data`, which will be called by the
    :class:`~composer.loggers.logger.Logger` whenever there is data to log.
    
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
        """Invoked by the :class:`~composer.loggers.logger.Logger` whenever there is a data to log.

        The logger callback should implement this method to log the data
        (e.g. write it to a file, send it to a server, etc...).

        .. note::

            This method will block the training loop. For optimal performance, it is recommended to deepcopy the
            ``data`` (e.g. ``copy.deepcopy(data)``), and store the copied data in queue. Then, either:

            *   Use background thread(s) or process(s) to read from this queue to perform any I/O.
            *   Batch multiple ``data``\\s together and flush periodically on events, such as
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
        allow_overwrite: bool,
    ):
        """Handle logging of a file artifact stored at ``file_path`` to an artifact named ``artifact_name``.

        This method is invoked by the :class:`~composer.core.logging.logger.Logger`.

        Implement this method to store the logged file.

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
            allow_overwrite (bool, optional): Whether to overwrite an existing artifact with the same ``artifact_name``.
                (default: ``False``)
        """
        del state, log_level, artifact_name, file_path, allow_overwrite  # unused
        pass
