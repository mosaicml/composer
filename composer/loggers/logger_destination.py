# Copyright 2021 MosaicML. All Rights Reserved.

"""Base class for logger callback."""

from __future__ import annotations

from abc import ABC

from composer.core.callback import Callback
from composer.core.state import State
from composer.core.time import Timestamp
from composer.loggers.logger import LoggerDataDict, LogLevel

__all__ = ["LoggerDestination"]


class LoggerDestination(Callback, ABC):
    """Base class for a logger destination. This is a :class:`~.Callback` with an additional interface for logging
    data, :meth:`log_data`. Custom loggers should extend this class. Data to be logged should be of the type
    :attr:`~.LoggerDataDict` (i.e. a ``{'name': value}`` mapping).

    For example, to define a custom logger and use it in training:

    .. code-block:: python

        from composer.loggers import LoggerDestination

        class MyLogger(LoggerDestination)

            def log_data(self, timestamp, log_level, data):
                print(f'Timestamp: {timestamp}: {log_level} {data}')

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_duration="1ep",
            optimizers=[optimizer],
            loggers=[MyLogger()]
        )
    """

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        """Called by the :class:`~composer.loggers.logger.Logger` to determine whether to log data given the ``log_level``.

        By default, it always returns ``True``, but this method
        can be overridden.
        Args:
            state (State): The global state object.
            log_level (LogLevel): The log level
        Returns:
            bool: Whether to log a data call, given the :class:`~.core.state.State` and :class:`~.LogLevel`.
        """
        del state, log_level  # unused
        return True

    def log_data(self, timestamp: Timestamp, log_level: LogLevel, data: LoggerDataDict):
        """Invoked by the :class:`~composer.loggers.logger.Logger` whenever there is a data to log.

        The logger callback should implement this method to log the data
        (e.g. write it to a file, send it to a server, etc...).

        .. note::

            This method will block the training loop. For optimal performance, it is recommended to
            ``copy.deepcopy(data)``, and store the copied data in queue. Background thread(s) or process(s) should
            read from this queue to perform any processing.

        Args:
            timestamp (Timestamp): The timestamp for the logged data.
            log_level (LogLevel): The log level.
            data (LoggerDataDict): The data to log.
        """
        del timestamp, log_level, data  # unused
        pass
