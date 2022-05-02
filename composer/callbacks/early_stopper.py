# Copyright 2021 MosaicML. All Rights Reserved.

"""Early stopping callback."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from composer.core import State
from composer.core.callback import Callback
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ["EarlyStopper"]


class EarlyStopper(Callback):
    __doc__ = f"""Callback to halt training early.
    """

    def __init__(
        self,
        monitor: str,
        eval_label: str,
        comp: Callable = None,
        min_delta: float = 0.0,
        patience: int = 1,
    ):
        self.monitor = monitor
        self.eval_label = eval_label
        self.comp = comp
        self.min_delta = abs(min_delta)
        if self.comp is None:
            self.comp = np.less if 'loss' in monitor.lower() or 'error' in monitor.lower() else np.greater
            if self.comp == np.less:
                self.min_delta *= -1

        if self.comp(1, 2):
            self.best = float('inf')
            self.min_delta *= -1
        else:
            self.best = -float('inf')

        self.patience = patience
        self.wait = 0

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.eval_label == "train":
            # if the monitored metric is an eval metric or in an evaluator
            return

        monitored_metric = None
        current_metrics = state.current_metrics
        if self.eval_label in current_metrics:
            if self.monitor in current_metrics[self.eval_label]:
                monitored_metric = current_metrics[self.eval_label][self.monitor]
            else:
                logger.warning(f"Couldn't find the metric {self.monitor} in the current_metrics/{self.eval_label}")
                return
        else:
            logger.warning(
                f"The label {self.eval_label} isn't in the state's current_metrics. Use the values train, eval, or the name of the Evaluator if using Evaluators."
            )
            return

        if self.comp(monitored_metric - self.min_delta, self.best):
            self.best = monitored_metric
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            # stop the training the training
            state.max_duration = state.timer.batch

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.eval_label != "train":
            # if the monitored metric is not an eval metric, the right logic is run on EPOCH_END
            return

        monitored_metric = None
        current_metrics = state.current_metrics
        if self.eval_label in current_metrics:
            if self.monitor in current_metrics[self.eval_label]:
                monitored_metric = current_metrics[self.eval_label][self.monitor]
            else:
                logger.warning(f"Couldn't find the metric {self.monitor} in the current_metrics/{self.eval_label}")
                return
        else:
            logger.warning(
                f"The label {self.eval_label} isn't in the state's current_metrics. Use the values train, eval, or the name of the Evaluator if using Evaluators."
            )
            return

        if self.comp(monitored_metric - self.min_delta, self.best):
            self.best = monitored_metric
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            # stop the training the training
            state.max_duration = state.timer
