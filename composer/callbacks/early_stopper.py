# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Early stopping callback."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['EarlyStopper']


class EarlyStopper(Callback):
    """Track a metric and halt training if it does not improve within a given interval.

    Example:
    .. doctest::

        >>> from composer import Evaluator, Trainer
        >>> from composer.callbacks.early_stopper import EarlyStopper
        >>> # constructing trainer object with this callback
        >>> early_stopper = EarlyStopper("Accuracy", "my_evaluator", patience=1)
        >>> evaluator = Evaluator(
        ...     dataloader = eval_dataloader,
        ...     label = 'my_evaluator',
        ...     metric_names = ['Accuracy']
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
        dataloader_label (str): The label of the dataloader or evaluator associated with the tracked metric.

            If ``monitor`` is in an :class:`.Evaluator`, the ``dataloader_label`` field should be set to the label of the
            :class:`.Evaluator`.

            If monitor is a training metric or an ordinary evaluation metric not in an :class:`.Evaluator`,
            the ``dataloader_label`` should be set to the dataloader label, which defaults to ``'train'`` or
            ``'eval'``, respectively.
        comp (str | (Any, Any) -> Any, optional): A comparison operator to measure change of the monitored metric.
            The comparison operator will be called ``comp(current_value, prev_best)``. For metrics where the optimal value is low
            (error, loss, perplexity), use a less than operator, and for metrics like accuracy where the optimal value
            is higher, use a greater than operator. Defaults to :func:`torch.less` if loss, error, or perplexity are substrings
            of the monitored metric, otherwise defaults to :func:`torch.greater`.
        min_delta (float, optional): An optional float that requires a new value to exceed the best value by at least that amount.
            Default: ``0.0``.
        patience (Time | int | str, optional): The interval of time the monitored metric can not improve without stopping
            training. Default: 1 epoch. If patience is an integer, it is interpreted as the number of epochs.
    """

    def __init__(
        self,
        monitor: str,
        dataloader_label: str,
        comp: Optional[Union[str, Callable[[
            Any,
            Any,
        ], Any]]] = None,
        min_delta: float = 0.0,
        patience: Union[int, str, Time] = 1,
    ):
        self.monitor = monitor
        self.dataloader_label = dataloader_label
        self.min_delta = abs(min_delta)
        if callable(comp):
            self.comp_func = comp
        if isinstance(comp, str):
            if comp.lower() in ('greater', 'gt'):
                self.comp_func = torch.greater
            elif comp.lower() in ('less', 'lt'):
                self.comp_func = torch.less
            else:
                raise ValueError(
                    "Unrecognized comp string. Use the strings 'gt', 'greater', 'lt' or 'less' or a callable comparison operator"
                )
        if comp is None:
            if any(substr in monitor.lower() for substr in ['loss', 'error', 'perplexity']):
                self.comp_func = torch.less
            else:
                self.comp_func = torch.greater

        self.best = None
        self.best_occurred = None

        if isinstance(patience, str):
            self.patience = Time.from_timestring(patience)
        elif isinstance(patience, int):
            self.patience = Time(patience, TimeUnit.EPOCH)
        else:
            self.patience = patience
            if self.patience.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
                raise ValueError('If `patience` is an instance of Time, it must have units of EPOCH or BATCH.')

    def _get_monitored_metric(self, state: State):
        if self.dataloader_label == 'train':
            if self.monitor in state.train_metrics:
                return state.train_metrics[self.monitor].compute()
        else:
            if self.monitor in state.eval_metrics[self.dataloader_label]:
                return state.eval_metrics[self.dataloader_label][self.monitor].compute()
        raise ValueError(f"Couldn't find the metric {self.monitor} with the dataloader label {self.dataloader_label}."
                         "Check that the dataloader_label is set to 'eval', 'train' or the evaluator name.")

    def _update_stopper_state(self, state: State):
        metric_val = self._get_monitored_metric(state)

        if not torch.is_tensor(metric_val):
            metric_val = torch.tensor(metric_val)

        if self.best is None:
            self.best = metric_val
            self.best_occurred = state.timestamp
        elif self.comp_func(metric_val, self.best) and torch.abs(metric_val - self.best) > self.min_delta:
            self.best = metric_val
            self.best_occurred = state.timestamp

        assert self.best_occurred is not None
        if self.patience.unit == TimeUnit.EPOCH:
            if state.timestamp.epoch - self.best_occurred.epoch > self.patience:
                state.stop_training()
        elif self.patience.unit == TimeUnit.BATCH:
            if state.timestamp.batch - self.best_occurred.batch > self.patience:
                state.stop_training()
        else:
            raise ValueError(f'The units of `patience` should be EPOCH or BATCH.')

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            # if the monitored metric is an eval metric or in an evaluator
            self._update_stopper_state(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            # if the monitored metric is not an eval metric, the right logic is run on EPOCH_END
            self._update_stopper_state(state)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.patience.unit == TimeUnit.BATCH and self.dataloader_label == state.dataloader_label:
            self._update_stopper_state(state)
