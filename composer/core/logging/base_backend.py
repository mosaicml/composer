# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import warnings
from abc import ABC
from typing import TYPE_CHECKING, List, Optional, Tuple, final

from composer.core.callback import Callback, RankZeroCallback
from composer.core.logging.logger import Logger
from composer.utils.ddp import is_rank_set, is_rank_zero

if TYPE_CHECKING:
    from composer.core.logging.logger import LogLevel, TLogData
    from composer.core.state import State


class DeferredLogMetricWarning(UserWarning):
    pass


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
    :func:`_will_log`, :func:`_log_metric`, and :func:`_training_start` instead of
    :func:`will_log`, :func:`log_metric`, and :func:`training_start`, respectively.

    This class ensures that :func:`_log_metric` and :func:`_training_start` are invoked only
    on the rank zero process.

    It caputres all logged data before the global rank is available.
    On the rank zero process, during the
    :attr:`~composer.core.event.Event.TRAINING_START` event (which occurs
    after the global rank is set), it routes all captured logged data to
    :func:`_log_metric`. For other processes, the captured log data
    is eventually discarded.

    .. automethod:: _will_log
    .. automethod:: _log_metric
    .. automethod:: _training_start
    """

    def __init__(self) -> None:
        super().__init__()
        # self._deferred_log_metric_calls is set to None once the logger is initialized
        self._deferred_log_metric_calls: Optional[List[Tuple[int, int, LogLevel, TLogData]]] = []

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
        if state.is_rank_set and not state.is_rank_zero:
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
        if is_rank_set() and not is_rank_zero():
            # no log if not on rank zero, clear deferred calls to free memory
            self._deferred_log_metric_calls = None
            return
        if self._deferred_log_metric_calls is not None:
            warnings.warn(
                f"{self.__class__.__name__}.log_metric() was invoked before training_start()."
                "This log call will be queued and processed after training_start().",
                category=DeferredLogMetricWarning)
            self._deferred_log_metric_calls.append((epoch, step, log_level, data))
            return
        return self._log_metric(epoch, step, log_level, data)

    def _training_start(self, state: State, logger: Logger) -> None:
        """Callback called on the
        :attr:`~composer.core.event.Event.TRAINING_START` event.

        Args:
            state (State): The global state.
            logger (Logger): The global logger.
        """
        del state, logger  # unused
        pass

    @final
    def training_start(self, state: State, logger: Logger) -> None:
        self._training_start(state, logger)  # initialize the logger
        if self._deferred_log_metric_calls is None:
            raise RuntimeError("_deferred_log_metric_calls should not be None")
        for epoch, step, log_level, data in self._deferred_log_metric_calls:
            self._log_metric(epoch, step, log_level, data)
        self._deferred_log_metric_calls = None
