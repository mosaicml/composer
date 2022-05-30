# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence

import torch

from composer.core import Callback, State, TimeUnit
from composer.loggers import Logger
from composer.trainer.devices.device import Device


class MetricSetterCallback(Callback):

    def __init__(self,
                 monitor: str,
                 dataloader_label: str,
                 metric_sequence: Sequence,
                 unit: TimeUnit,
                 device: Optional[Device] = None):
        self.monitor = monitor
        self.dataloader_label = dataloader_label
        self.metric_sequence = metric_sequence
        self.unit = unit
        self.device = device

    def _update_metrics(self, state: State):
        idx = min(len(self.metric_sequence) - 1, state.timestamp.get(self.unit).value)
        metric_val = self.metric_sequence[idx]
        state.current_metrics[self.dataloader_label] = state.current_metrics.get(self.dataloader_label, dict())
        metric_tensor = torch.tensor(metric_val)
        if self.device is not None:
            self.device.tensor_to_device(metric_tensor)
        state.current_metrics[self.dataloader_label][self.monitor] = metric_tensor

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            self._update_metrics(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            self._update_metrics(state)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.unit == TimeUnit.BATCH and self.dataloader_label == state.dataloader_label:
            self._update_metrics(state)
