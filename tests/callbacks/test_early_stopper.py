from types import MethodType
from typing import List

import numpy as np
import pytest
from torch import tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection

from composer import Trainer
from composer.callbacks.early_stopper import EarlyStopper
from composer.core.evaluator import Evaluator
from composer.loggers import LogLevel
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.parametrize('metric_sequence', [[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.2]])
def test_threshold_stopper(metric_sequence: List[int]):

    early_stopper = EarlyStopper("Accuracy", "eval", patience=3)

    def test_compute_and_log_metrics(self, dataloader_label: str, log_level: LogLevel, metrics: MetricCollection):
        idx = min(len(metric_sequence) - 1, self.state.timer.epoch.value)
        metric_val = metric_sequence[idx]
        self.state.current_metrics[dataloader_label] = {"Accuracy": tensor(metric_val)}

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
        callbacks=[early_stopper],
    )

    trainer._compute_and_log_metrics = MethodType(test_compute_and_log_metrics, trainer)

    trainer.fit()

    assert trainer.state.timer.epoch.value == len(metric_sequence) + early_stopper.patience - 1


def test_threshold_stopper_evaluator():
    metric_sequence = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    early_stopper = EarlyStopper("Accuracy", "evaluator_1", patience=3)

    def test_compute_and_log_metrics(self, dataloader_label: str, log_level: LogLevel, metrics: MetricCollection):
        idx = min(len(metric_sequence) - 1, self.state.timer.epoch.value)
        metric_val = metric_sequence[idx]
        self.state.current_metrics[dataloader_label] = {"Accuracy": tensor(metric_val)}

    evaluator = Evaluator(label="evaluator_1",
                          dataloader=DataLoader(
                              RandomClassificationDataset(shape=(5, 1, 1)),
                              batch_size=4,
                          ),
                          metrics=Accuracy())

    trainer = Trainer(
        model=SimpleModel(num_features=5),
        train_dataloader=DataLoader(
            RandomClassificationDataset(shape=(5, 1, 1)),
            batch_size=4,
        ),
        eval_dataloader=evaluator,
        max_duration="30ep",
        callbacks=[early_stopper],
    )

    trainer._compute_and_log_metrics = MethodType(test_compute_and_log_metrics, trainer)

    trainer.fit()

    assert trainer.state.timer.epoch.value == len(metric_sequence) + early_stopper.patience - 1
