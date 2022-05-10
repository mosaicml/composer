from typing import List, Optional, Sequence

import pytest
from torch import tensor
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks.early_stopper import EarlyStopper
from composer.core import State
from composer.core.callback import Callback
from composer.core.time import Time, TimeUnit
from composer.loggers import Logger
from composer.trainer.devices.device import Device
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.devices.device_gpu import DeviceGPU
from tests.common import RandomClassificationDataset, SimpleModel, device


class TestMetricSetter(Callback):

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
        metric_tensor = tensor(metric_val)
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


@device('cpu', 'gpu')
@pytest.mark.timeout(5)
@pytest.mark.parametrize('metric_sequence', [[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3], [0.1, 0.2]])
@pytest.mark.parametrize('unit', [TimeUnit.EPOCH, TimeUnit.BATCH])
def test_early_stopper(metric_sequence: List[float], unit: TimeUnit, device: str):

    if unit == TimeUnit.EPOCH:
        dataloader_label = "eval"
    else:
        dataloader_label = "train"

    test_device = DeviceGPU() if device == 'gpu' else DeviceCPU()

    early_stopper = EarlyStopper("Accuracy", dataloader_label, patience=Time(3, unit))

    test_metric_setter = TestMetricSetter("Accuracy", dataloader_label, metric_sequence, unit, test_device)

    trainer = Trainer(
        model=SimpleModel(num_features=5),
        train_dataloader=DataLoader(
            RandomClassificationDataset(shape=(5, 1, 1)),
            batch_size=4,
        ),
        eval_dataloader=DataLoader(
            RandomClassificationDataset(shape=(5, 1, 1)),
            batch_size=4,
        ),
        max_duration="30ep",
        callbacks=[test_metric_setter, early_stopper],
    )

    trainer.fit()

    assert trainer.state.timestamp.get(unit).value == len(metric_sequence) + int(early_stopper.patience)
