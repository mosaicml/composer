from typing import List, Sequence

import pytest
from torch import tensor
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks.early_stopper import EarlyStopper
from composer.core import State
from composer.core.callback import Callback
from composer.loggers import Logger
from tests.common import RandomClassificationDataset, SimpleModel


class TestMetricSetter(Callback):

    def __init__(self, monitor: str, dataloader_label: str, metric_sequence: Sequence):
        self.monitor = monitor
        self.dataloader_label = dataloader_label
        self.metric_sequence = metric_sequence

    def _update_metrics(self, state: State):
        idx = min(len(self.metric_sequence) - 1, state.timer.epoch.value)
        metric_val = self.metric_sequence[idx]
        state.current_metrics[self.dataloader_label][self.monitor] = tensor(metric_val)

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label != "train":
            self._update_metrics(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == "train":
            self._update_metrics(state)


@pytest.mark.parametrize('metric_sequence', [[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3], [0.1, 0.2]])
def test_threshold_stopper(metric_sequence: List[float]):

    early_stopper = EarlyStopper("Accuracy", "eval", patience=3)

    test_metric_setter = TestMetricSetter("Accuracy", "eval", metric_sequence)

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

    assert trainer.state.timer.epoch.value == len(metric_sequence) + early_stopper.patience
