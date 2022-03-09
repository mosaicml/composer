# Copyright 2021 MosaicML. All Rights Reserved.

"""Base class for logger callback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from composer.core.callback import Callback
from composer.core import State

if TYPE_CHECKING:
    from composer.core.logging.logger import LoggerDataDict, LogLevel

__all__ = ["LoggerDestination"]


class LoggerDestination(Callback, ABC):
    """Base class for logger destination.

    Subclasses must implement :meth:`log_data`, which will be called by the
    :class:`~composer.core.logging.logger.Logger` whenever there is data to log.
    
    As this class extends :class:`~.callback.Callback`, logger destinations can run on any training loop
    :class:`~composer.core.event.Event`. For example, it may be helpful to run on
    :attr:`~composer.core.event.Event.EPOCH_END` to perform any flushing at the end of every epoch.

    Example
    -------

    >>> from composer.core.logging import LoggerDestination
    >>> class MyLogger(LoggerDestination):
    ... def log_data(self, timestamp, log_level, data):
    ...     print(f'Timestamp: {timestamp}: {log_level} {data}')
    >>> trainer = Trainer(
    ...     ...,
    ...     logger_destinations=[MyLogger()]
    ... )
    """

    @abstractmethod
    def log_data(self, state: State, log_level: LogLevel, data: LoggerDataDict):
        """Invoked by the :class:`~composer.core.logging.logger.Logger` whenever there is a data to log.

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
        pass
