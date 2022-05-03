# Copyright 2021 MosaicML. All Rights Reserved.

"""Early stopping callback."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

import torch

from composer.core import State, Time
from composer.core.callback import Callback
from composer.core.time import TimeUnit
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ["EarlyStopper"]


class EarlyStopper(Callback):
    """This callback tracks a training or evaluation metric and halts training if the metric does not
    improve within a given interval.

    Example

    .. doctest::

        >>> from composer.callbacks.early_stopper import EarlyStopper
        >>> from torchmetrics.classification.accuracy import Accuracy
        >>> # constructing trainer object with this callback
        >>> early_stopper = EarlyStopper("Accuracy", "my_evaluator", patience=1)
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
        ...     callbacks=[early_stopper],
        ... )

    Args:
        monitor (str): The name of the metric to monitor.
        dataloader_label (str): The label of the dataloader or evaluator associated with the tracked metric. If 
            monitor is in an Evaluator, the dataloader_label field should be set to the Evaluator's label. If 
            monitor is a training metric or an ordinary evaluation metric not in an Evaluator, dataloader_label
            should be set to 'train' or 'eval' respectively.
        comp (Callable[[Any, Any], bool], optional): A comparison operator to measure change of the monitored metric. The comparison
            operator will be called ``comp(current_value, prev_best)``. For metrics where the optimal value is low
            (error, loss, perplexity), use a less than operator and for metrics like accuracy where the optimal value
            is higher, use a greater than operator. Defaults to :func:`torch.less` if loss, error, or perplexity are substrings
            of the monitored metric, otherwise defaults to :func:`torch.greater`
        min_delta (float, optional): An optional float that requires a new value to exceed the best value by at
            least that amount. Defaults to 0.
        patience (int | str | Time, optional): The interval of time the monitored metric can not improve without stopping
            training. Defaults to 1 epoch. If patience is an integer, it is interpreted as the number of epochs.
    """

    def __init__(
        self,
        monitor: str,
        dataloader_label: str,
        comp: Optional[Callable[[
            Any,
            Any,
        ], bool]] = None,
        min_delta: float = 0.0,
        patience: Union[int, str, Time] = 1,
    ):
        self.monitor = monitor
        self.dataloader_label = dataloader_label
        self.comp = comp
        self.min_delta = abs(min_delta)
        if self.comp is None:
            if any(substr in monitor.lower() for substr in ["loss", "error", "perplexity"]):
                self.comp = torch.less
            else:
                self.comp = torch.greater

        self.best = None
        self.best_occurred = None

        if isinstance(patience, str):
            self.patience = Time.from_timestring(patience)
        elif isinstance(patience, int):
            self.patience = Time(patience, TimeUnit.EPOCH)
        else:
            self.patience = patience
            if self.patience.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
                raise ValueError("If `patience` is an instance of Time, it must have units of EPOCH or BATCH.")

    def _get_monitored_metric(self, state: State):
        if self.dataloader_label in state.current_metrics:
            if self.monitor in state.current_metrics[self.dataloader_label]:
                return state.current_metrics[self.dataloader_label][self.monitor]
        raise ValueError(f"Couldn't find the metric {self.monitor} with the dataloader label {self.dataloader_label}."
                         "Check that the dataloader_label is set to 'eval', 'train' or the evaluator name.")

    def _update_stopper_state(self, state: State):
        metric_val = self._get_monitored_metric(state)

        if not torch.is_tensor(metric_val):
            metric_val = torch.tensor(metric_val)

        assert self.comp is not None
        if self.best is None:
            self.best = metric_val
            self.best_occurred = state.timer.get_timestamp()
        elif self.comp(metric_val, self.best) and torch.abs(metric_val - self.best) > self.min_delta:
            self.best = metric_val
            self.best_occurred = state.timer.get_timestamp()

        assert self.best_occurred is not None
        if self.patience.unit == TimeUnit.EPOCH:
            if state.timer.epoch - self.best_occurred.epoch > self.patience:
                state.max_duration = state.timer.batch
        elif self.patience.unit == TimeUnit.BATCH:
            if state.timer.batch - self.best_occurred.batch > self.patience:
                state.max_duration = state.timer.batch
        else:
            raise ValueError(f"The units of `patience` should be EPOCH or BATCH.")

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label != "train":
            # if the monitored metric is an eval metric or in an evaluator
            self._update_stopper_state(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == "train":
            # if the monitored metric is not an eval metric, the right logic is run on EPOCH_END
            self._update_stopper_state(state)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.patience.unit == TimeUnit.BATCH and self.dataloader_label == "train":
            self._update_stopper_state(state)
