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
    """BaseLoggerBackend defines the interface for logging destinations."""

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        """
        will_log() returns whether the logging destination will log a metric
        or state change, given current state `state` and level `log_level`
        """
        return True

    def log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData):
        """
        log_metric() is called whenever there is a metric to log, for which `self.will_log(state, log_level)`
        returned True.
        """
        pass


class RankZeroLoggerBackend(BaseLoggerBackend, RankZeroCallback, ABC):

    def __init__(self) -> None:
        super().__init__()
        # self.deferred_log_metric_calls is set to None once the logger is initialized
        self.deferred_log_metric_calls: Optional[List[Tuple[int, int, LogLevel, TLogData]]] = []

    def _will_log(self, state: State, log_level: LogLevel) -> bool:
        """
        _will_log() returns whether the logging destination will log a metric
        or state change, given current state `state` and level `log_level`
        """
        return True

    @final
    def will_log(self, state: State, log_level: LogLevel) -> bool:
        if state.is_rank_set and not state.is_rank_zero:
            return False
        return self._will_log(state, log_level)

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData) -> None:
        """
        _log_metric() is called whenever there is a metric to log, for which `self.will_log(state, log_level)`
        returned True.
        """
        pass

    @final
    def log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData) -> None:
        if is_rank_set() and not is_rank_zero():
            # definitely not interested in the log message
            # lazily clear all deferred calls that will not be logged
            # so the memory can be freed
            self.deferred_log_metric_calls = None
            return
        if self.deferred_log_metric_calls is not None:
            warnings.warn(
                f"{self.__class__.__name__}.log_metric() was invoked before training_start()."
                "This log call will be queued and processed after training_start().",
                category=DeferredLogMetricWarning)
            self.deferred_log_metric_calls.append((epoch, step, log_level, data))
            return
        return self._log_metric(epoch, step, log_level, data)

    def _training_start(self, state: State, logger: Logger) -> None:
        pass

    @final
    def training_start(self, state: State, logger: Logger) -> None:
        self._training_start(state, logger)  # initialize the logger
        if self.deferred_log_metric_calls is None:
            raise RuntimeError("deferred_log_metric_calls should not be None")
        for epoch, step, log_level, data in self.deferred_log_metric_calls:
            self._log_metric(epoch, step, log_level, data)
        self.deferred_log_metric_calls = None
