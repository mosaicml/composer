# Copyright 2021 MosaicML. All Rights Reserved.

"""Base class for logger callback."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from composer.core.callback import Callback
from composer.core.time import Timestamp

if TYPE_CHECKING:
    from composer.core.logging.logger import LogLevel, TLogData
    from composer.core.state import State

__all__ = ["LoggerCallback"]


class LoggerCallback(Callback, ABC):
    """Base class for logger callback. This is a :class:`~.callback.Callback` with an additional interface for logging
    metrics, :func:`log_metric`. Custom loggers should extend this class. Data to be logged should be of the type
    :attr:`~.logger.TLogData` (i.e. a ``{'name': value}`` mapping).

    For example, to define a custom logger and use it in training:

    .. code-block:: python

        from composer.core.logging import LoggerCallback

        class MyLogger(LoggerCallback)

            def log_metric(self, timestamp, log_level, data):
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

    def __init__(self):
        super().__init__()

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        """Called by the :class:`~.logging.logger.Logger` to determine whether to log a metric.

        By default, it always returns ``True``, but this method
        can be overridden.

        Args:
            state (State): The global state object.
            log_level (LogLevel): The log level

        Returns:
            bool: Whether to log a metric call, given the
                :class:`~.core.state.State` and
                :class:`~.logging.logger.LogLevel`.
        """
        del state, log_level  # unused
        return True

    def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData):
        """Called by the :class:`~.logging.logger.Logger` for metrics where :func:`will_log` returned ``True``.

        The logger callback should override this function to log the data
        (e.g. write it to a file, send it to a server, etc...).

        Args:
            timestamp (Timestamp): The timestamp for the logged data.
            log_level (LogLevel): The log level.
            data (TLogData): The metric to log.
        """
        del timestamp, log_level, data  # unused
        pass
