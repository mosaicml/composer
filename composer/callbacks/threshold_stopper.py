from typing import Any, Callable

import numpy as np

from composer import Callback
from composer.core import State
from composer.loggers import Logger


class ThresholdStopper(Callback):
    """This callback tracks a training or evaluation metric and halts training when the 
    metric value reaches a certain threshold.

    Example

    .. doctest::

        >>> from composer.callbacks.threshold_stopper import ThresholdStopper
        >>> from torchmetrics.classification.accuracy import Accuracy
        >>> # constructing trainer object with this callback
        >>> threshold_stopper = ThresholdStopper("Accuracy", "my_evaluator", 0.7)
        >>> evaluator = Evaluator(
        ...     dataloader = eval_dataloader,
        ...     label = 'my_evaluator',
        ...     metrics = Accuracy()
        ... )
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     eval_dataloader=evaluator,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[threshold_stopper],
        ... )

    Args:
        monitor (str): The name of the metric to monitor.
        dataloader_label (str): The label of the dataloader or evaluator associated with the tracked metric. If 
            monitor is in an Evaluator, the dataloader_label field should be set to the Evaluator's label. If 
            monitor is a training metric or an ordinary evaluation metric not in an Evaluator, dataloader_label
            should be set to 'train' or 'eval' respectively. If dataloader_label is set to 'train', then the 
            callback will stop training in the middle of the epoch.
        threshold (float): The threshold that dictates when to halt training. Whether training stops if the metric
            exceeds or falls below the threshold depends on the comparison operator.
        comp (Callable[[Any, Any], bool], optional): A comparison operator to measure change of the monitored metric.
            The comparison operator will be called as `comp(current_value, threshold)`. For metrics where the optimal 
            value is low (error, loss, perplexity), use a less than operator and for metrics like accuracy where the optimal
            value is higher, use a greater than operator. Defaults to numpy.less if loss, error, or perplexity are substrings
            of the monitored metric, otherwise defaults to numpy.greater.
    """

    def __init__(
        self,
        monitor: str,
        dataloader_label: str,
        threshold: float,
        comp: Callable[[Any, Any], bool] = None,
    ):
        self.monitor = monitor
        self.threshold = threshold
        self.comp = comp
        self.dataloader_label = dataloader_label
        if self.comp is None:
            if any(substr in monitor.lower() for substr in ["loss", "error", "perplexity"]):
                self.comp = np.less
            else:
                self.comp = np.greater

    def _get_monitored_metric(self, state: State):
        if self.dataloader_label in state.current_metrics:
            if self.monitor in state.current_metrics[self.dataloader_label]:
                return state.current_metrics[self.dataloader_label][self.monitor]
        raise ValueError(f"Couldn't find the metric {self.monitor} with the dataloader label {self.dataloader_label}."
                         "Check that the dataloader_label is set to 'eval', 'train' or the evaluator name.")

    def _compare_metric_and_update_state(self, state: State):
        metric_val = self._get_monitored_metric(state)
        if self.comp(metric_val, self.threshold):
            state.max_duration = state.timer.batch

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == "train":
            self._compare_metric_and_update_state(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == "train":
            self._compare_metric_and_update_state(state)

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label != "train":
            self._compare_metric_and_update_state(state)
