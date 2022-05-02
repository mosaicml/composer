from types import MethodType

import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection

from composer import Trainer
from composer.callbacks.threshold_stopper import ThresholdStopper
from composer.core.evaluator import Evaluator
from composer.loggers import LogLevel
from tests.common import SimpleModel
from tests.common.datasets import RandomClassificationDataset


def test_threshold_stopper_eval():
    metric_min = 0.4
    metric_max = 0.8
    metric_threshold = 0.75
    accuracy_sequence = (i for i in np.linspace(metric_min, metric_max, 100))

    tstop = ThresholdStopper("Accuracy", metric_threshold, "eval")

    def test_compute_and_log_metrics(self, dataloader_label: str, log_level: LogLevel, metrics: MetricCollection):
        metrics_val = next(accuracy_sequence, metric_max)
        self.state.current_metrics[dataloader_label] = {"Accuracy": tensor(metrics_val)}

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
        max_duration="10ep",
        callbacks=[tstop],
    )

    trainer._compute_and_log_metrics = MethodType(test_compute_and_log_metrics, trainer)

    trainer.fit()

    assert trainer.state.timer.epoch.value == 4


def test_threshold_stopper_evaluators():
    metric_min = 0.4
    metric_max = 0.8
    metric_threshold = 0.75
    accuracy_sequence = (i for i in np.linspace(metric_min, metric_max, 100))

    tstop = ThresholdStopper("Accuracy", metric_threshold, "evaluator_1")

    def test_compute_and_log_metrics(self, dataloader_label: str, log_level: LogLevel, metrics: MetricCollection):
        metrics_val = next(accuracy_sequence, metric_max)
        self.state.current_metrics[dataloader_label] = {"Accuracy": tensor(metrics_val)}

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
        max_duration="10ep",
        callbacks=[tstop],
    )

    trainer._compute_and_log_metrics = MethodType(test_compute_and_log_metrics, trainer)

    trainer.fit()

    assert trainer.state.timer.epoch.value == 4
