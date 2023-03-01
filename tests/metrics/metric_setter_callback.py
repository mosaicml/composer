# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from composer.core import Callback, State, TimeUnit
from composer.devices import Device
from composer.loggers import Logger


class MetricSetterCallback(Callback):

    def __init__(
            self,
            monitor: str,
            dataloader_label: str,
            metric_cls: Callable,  # metric function
            metric_sequence: Sequence,
            unit: TimeUnit,
            device: Optional[Device] = None,
            metric_args: Optional[Dict] = None):
        self.monitor = monitor
        self.dataloader_label = dataloader_label
        self.metric_cls = metric_cls
        self.metric_sequence = metric_sequence
        self.unit = unit
        self.device = device
        self.metric_args = metric_args
        if self.metric_args is None:
            self.metric_args = {}

    def _generate_dummy_metric_inputs(self, target_val) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake set of predictions and target values to satisfy the given target accuracy value."""
        # predictions is a tensor with a ratio of target_val 1s to sub_target 0s
        preds_ones = torch.ones(int(target_val * 10), dtype=torch.uint8)
        sub_target = float('{0:.2f}'.format((1 - target_val) * 10))
        preds_zeros = torch.zeros(int(sub_target), dtype=torch.uint8)
        preds = torch.cat((preds_ones, preds_zeros))
        # targets is a tensor full of ones
        targets = torch.ones(10, dtype=torch.uint8)
        return (preds, targets)

    def _update_metrics(self, state: State):
        idx = min(len(self.metric_sequence) - 1, state.timestamp.get(self.unit).value)
        metric_val = self.metric_sequence[idx]
        if self.dataloader_label == 'train':
            state.train_metrics = state.train_metrics if state.train_metrics else {}
        else:
            state.eval_metrics[self.dataloader_label] = state.eval_metrics.get(self.dataloader_label, dict())
        metric_tensor = torch.tensor(metric_val)
        if self.device is not None:
            self.device.tensor_to_device(metric_tensor)

        raw_metric = self.metric_cls(**self.metric_args)  # type: ignore
        preds, targets = self._generate_dummy_metric_inputs(metric_val)
        raw_metric.update(preds=preds, target=targets)

        # assert for pyright error: "module_to_device" is not a known member of "None"
        assert self.device is not None
        self.device.module_to_device(raw_metric)
        if self.dataloader_label == 'train':
            state.train_metrics[self.monitor] = raw_metric
        else:
            state.eval_metrics[self.dataloader_label][self.monitor] = raw_metric

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            self._update_metrics(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            self._update_metrics(state)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.unit == TimeUnit.BATCH and self.dataloader_label == state.dataloader_label:
            self._update_metrics(state)
