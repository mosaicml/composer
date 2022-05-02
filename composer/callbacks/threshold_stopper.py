from typing import Callable

import numpy as np

from composer import Callback
from composer.core import State
from composer.loggers import Logger


class ThresholdStopper(Callback):

    def __init__(
        self,
        monitor: str,
        threshold: float,
        eval_label: str,
        comp: Callable = None,
    ):
        self.monitor = monitor
        self.threshold = threshold
        self.comp = comp
        self.eval_label = eval_label

        if self.comp is None:
            self.comp = np.less if 'loss' in monitor.lower() or 'error' in monitor.lower() else np.greater

    def epoch_end(self, state: State, logger: Logger) -> None:
        if not self.eval_label == "train":
            # if the monitored metric is an eval metric, wait for the event EVAL_END
            return

        # get monitored_val
        current_metrics = state.current_metrics
        if self.monitor in current_metrics["train"]:
            monitored_metric = current_metrics["train"][self.monitor]
        else:
            logger.warning(f"Couldn't find the training metric {self.monitor}."
                           "Check that eval_label is correctly set to 'train', 'eval', or the evaluator name")
            return

        if self.comp(monitored_metric, self.threshold):
            state.max_duration = state.timer.batch

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.eval_label == "train":
            # if the monitored metric is not an eval metric, the right logic is run on EPOCH_END
            return

        # get monitored_val
        current_metrics = state.current_metrics
        if "eval" in current_metrics:
            if self.monitor in current_metrics["eval"]:
                monitored_metric = current_metrics["eval"][self.monitor]
            else:
                logger.warning(f"Couldn't find the metric {self.monitor} in the 'eval' metrics."
                               "Check the eval_label field to make sure it's correct.")
                return
        elif self.eval_label in current_metrics:
            if self.monitor in current_metrics[self.eval_label]:
                monitored_metric = current_metrics[self.eval_label][self.monitor]
        else:
            logger.warning(
                f"Couldn't find the training metric {self.monitor}. Check the eval_label field to make sure it's a valid evaluator."
            )
            return

        if self.comp(monitored_metric, self.threshold):
            state.max_duration = state.timer.batch
