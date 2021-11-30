# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from composer.core.callback import Callback, RankZeroCallback
from composer.utils import ddp

if TYPE_CHECKING:
    from composer.core.logging.logger import LogLevel, TLogData
    from composer.core.state import State

try:
    from typing import final
except ImportError:
    final = lambda x: x  # final is not available in python 3.7


class BaseLoggerBackend(Callback, ABC):
    """Base class for logging backends.
    """

    def __init__(self):
        super().__init__()

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        """Called by the :class:`~composer.core.logging.logger.Logger`
        to determine whether to log a metric.

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

    def log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData):
        """Called by the :class:`~composer.core.logging.logger.Logger`
        for metrics where :func:`will_log` returned ``True``.

        The logging backend should override this function to log the data
        (e.g. write it to a file, send it to a server, etc...).

        Args:
            epoch (int): The epoch for the logged data.
            step (int): The global step for the logged data.
            log_level (LogLevel): The log level.
            data (TLogData): The metric to log.
        """
        del epoch, step, log_level, data  # unused
        pass


class RankZeroLoggerBackend(BaseLoggerBackend, RankZeroCallback, ABC):
    """Base class for logging backends that run only on the rank zero process.

    In a multi-process training setup (e.g. when using DistributedDataParallel),
    some logging backends require that only the rank zero process log data.
    For example, when logging to a file, only the main process should open the file
    and save data.

    When using this class, override
    :func:`_will_log` and :func:`_log_metric`` instead of
    :func:`will_log` and :func:`log_metric`, respectively.

    This class ensures that :func:`_will_log` and :func:`_log_metric`
    are invoked only on the rank zero process.

    .. automethod:: _will_log
    .. automethod:: _log_metric
    """

    def __init__(self) -> None:
        super().__init__()

    def _will_log(self, state: State, log_level: LogLevel) -> bool:
        """Called by the :class:`~composer.core.logging.logger.Logger`
        to determine whether the logging backend will log a metric.

        By default, it always returns ``True``, but this method
        can be overridden.

        Args:
            state (State): The global state object.
            log_level (LogLevel): The log level.

        Returns:
            bool: Whether to log a metric call, given the
            :class:`~composer.core.state.State` and
            :class:`~composer.core.logging.logger.LogLevel`.
        """
        del state, log_level  # Unused
        return True

    @final
    def will_log(self, state: State, log_level: LogLevel) -> bool:
        if ddp.get_local_rank() != 0:
            return False
        return self._will_log(state, log_level)

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData) -> None:
        """Called by the :class:`~composer.core.logging.logger.Logger`
        for metrics where :func:`will_log` returned ``True``.

        The logging backend should override this function to log the data
        (e.g. write it to a file, send it to a server, etc...).

        Args:
            epoch (int): The epoch for the logged data.
            step (int): The global step for the logged data.
            log_level (LogLevel). The log level.
            data (TLogData): The metric to log.
        """
        del epoch, step, log_level, data  # Unused
        pass

    @final
    def log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData) -> None:
        if ddp.get_local_rank() != 0:
            return
        return self._log_metric(epoch, step, log_level, data)
