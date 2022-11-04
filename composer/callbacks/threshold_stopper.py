# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold stopping callback."""

from typing import Any, Callable, Optional, Union

import torch

from composer.core import Callback, State
from composer.loggers import Logger


class ThresholdStopper(Callback):
    """Halt training when a metric value reaches a certain threshold.

    Example:
        .. doctest::

            >>> from composer.callbacks.threshold_stopper import ThresholdStopper
            >>> from torchmetrics.classification.accuracy import Accuracy
            >>> # constructing trainer object with this callback
            >>> threshold_stopper = ThresholdStopper("Accuracy", "my_evaluator", 0.7)
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
        comp (Callable[[Any, Any], Any], optional): A comparison operator to measure change of the monitored metric. The comparison
            operator will be called ``comp(current_value, prev_best)``. For metrics where the optimal value is low
            (error, loss, perplexity), use a less than operator and for metrics like accuracy where the optimal value
            is higher, use a greater than operator. Defaults to :func:`torch.less` if loss, error, or perplexity are substrings
            of the monitored metric, otherwise defaults to :func:`torch.greater`
        stop_on_batch (bool, optional): A bool that indicates whether to stop training in the middle of an epoch if
            the training metrics satisfy the threshold comparison. Defaults to False.
    """

    def __init__(self,
                 monitor: str,
                 dataloader_label: str,
                 threshold: float,
                 *,
                 comp: Optional[Union[str, Callable[[
                     Any,
                     Any,
                 ], Any]]] = None,
                 stop_on_batch: bool = False):
        self.monitor = monitor
        self.threshold = threshold
        self.dataloader_label = dataloader_label
        self.stop_on_batch = stop_on_batch
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

    def _get_monitored_metric(self, state: State):
        if self.dataloader_label == 'train':
            if self.monitor in state.train_metrics:
                return state.train_metrics[self.monitor].compute()
        else:
            if self.monitor in state.eval_metrics[self.dataloader_label]:
                return state.eval_metrics[self.dataloader_label][self.monitor].compute()
        raise ValueError(f"Couldn't find the metric {self.monitor} with the dataloader label {self.dataloader_label}."
                         "Check that the dataloader_label is set to 'eval', 'train' or the evaluator name.")

    def _compare_metric_and_update_state(self, state: State):
        metric_val = self._get_monitored_metric(state)

        if not torch.is_tensor(metric_val):
            metric_val = torch.tensor(metric_val)

        if self.comp_func(metric_val, self.threshold):
            state.stop_training()

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            # if the monitored metric is an eval metric or in an evaluator
            self._compare_metric_and_update_state(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            # if the monitored metric is not an eval metric, the right logic is run on EPOCH_END
            self._compare_metric_and_update_state(state)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.stop_on_batch and self.dataloader_label == state.dataloader_label:
            self._compare_metric_and_update_state(state)
