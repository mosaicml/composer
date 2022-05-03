from typing import List, Sequence

import pytest
from torch import tensor
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks.threshold_stopper import ThresholdStopper
from composer.core import State
from composer.core.callback import Callback
from composer.core.time import TimeUnit
from composer.loggers import Logger
from tests.common import RandomClassificationDataset, SimpleModel


class TestMetricSetter(Callback):

    def __init__(self, monitor: str, dataloader_label: str, metric_sequence: Sequence, unit: TimeUnit):
        self.monitor = monitor
        self.dataloader_label = dataloader_label
        self.metric_sequence = metric_sequence
        self.unit = unit

    def _update_metrics(self, state: State):
        idx = min(len(self.metric_sequence) - 1, state.timer.get(self.unit).value)
        metric_val = self.metric_sequence[idx]
        state.current_metrics[self.dataloader_label] = state.current_metrics.get(self.dataloader_label, dict())
        state.current_metrics[self.dataloader_label][self.monitor] = tensor(metric_val)

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label != "train":
            self._update_metrics(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == "train":
            self._update_metrics(state)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.unit == TimeUnit.BATCH:
            self._update_metrics(state)


@pytest.mark.parametrize('metric_sequence', [[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8], [0.6, 0.7]])
@pytest.mark.parametrize('unit', [TimeUnit.EPOCH, TimeUnit.BATCH])
def test_threshold_stopper_eval(metric_sequence: List[float], unit: TimeUnit):
    metric_threshold = 0.65

    if unit == TimeUnit.EPOCH:
        dataloader_label = "eval"
    else:
        dataloader_label = "train"

    tstop = ThresholdStopper("Accuracy", dataloader_label, metric_threshold)

    test_metric_setter = TestMetricSetter("Accuracy", dataloader_label, metric_sequence, unit)

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
        callbacks=[test_metric_setter, tstop],
    )

    trainer.fit()

    count_before_threshold = 0
    for metric in metric_sequence:
        if metric_threshold > metric:
            count_before_threshold += 1

    assert trainer.state.timer.get(unit).value == count_before_threshold
