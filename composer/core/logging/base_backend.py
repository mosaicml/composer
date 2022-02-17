# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from composer.core.callback import Callback
from composer.core.time import Timestamp

if TYPE_CHECKING:
    from composer.core.logging.logger import LogLevel, TLogData
    from composer.core.state import State


class LoggerCallback(Callback, ABC):
    """Base class for logging backends."""

    def __init__(self):
        super().__init__()

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        """Called by the :class:`~composer.core.logging.logger.Logger` to determine whether to log a metric.

        By default, it always returns ``True``, but this method
        can be overridden.

        Args:
            state (State): The global state object.
            log_level (LogLevel): The log level

        Returns:
            bool: Whether to log a metric call, given the
            :class:`~composer.core.state.State` and
            :class:`~composer.core.logging.logger.LogLevel`.
        """
        del state, log_level  # unused
        return True

    def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData):
        """Called by the :class:`~composer.core.logging.logger.Logger` for metrics where :func:`will_log` returned
        ``True``.

        The logging backend should override this function to log the data
        (e.g. write it to a file, send it to a server, etc...).

        Args:
            epoch (int): The epoch for the logged data.
            step (int): The global step for the logged data.
            log_level (LogLevel): The log level.
            data (TLogData): The metric to log.
        """
        del timestamp, log_level, data  # unused
        pass
