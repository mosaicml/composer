import os
from typing import Any, Dict, Optional

import pytest

import composer
from composer.core.logging.base_backend import RankZeroLoggerBackend
from composer.core.logging.logger import LogLevel, TLogData
from composer.loggers.logger_hparams import BaseLoggerBackendHparams, WandBLoggerBackendHparams
from composer.trainer import TrainerHparams

pytestmark = pytest.mark.regression


class MetricMonitor(RankZeroLoggerBackend):

    def __init__(self) -> None:
        super().__init__()
        self.metric_to_latest_value = {}

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                self.metric_to_latest_value[k] = v


class MetricMonitorHparams(BaseLoggerBackendHparams):

    def initialize_object(self, config: Optional[Dict[str, Any]] = None) -> MetricMonitor:
        return MetricMonitor()


TrainerHparams.register_class("loggers", MetricMonitorHparams, "metric_monitor")

EXPECTED_METRICS = {"resnet50": {"acc/val": [0.75, 0.79]}, "classify_mnist_cpu": {"acc/val": [0.98, 0.99]}}


@pytest.mark.timeout(0)  # disable timeouts
@pytest.mark.parametrize("model_name", [
    pytest.param("resnet50", marks=pytest.mark.n_gpus(8)),
    "classify_mnist_cpu",
])
def test_regression(model_name: str, is_main_pytest_process: bool):
    trainer_hparams = TrainerHparams.create(
        os.path.join(os.path.dirname(composer.__file__), "yamls", "models", f"{model_name}.yaml"))
    trainer_hparams.loggers = [
        MetricMonitorHparams(),
        WandBLoggerBackendHparams(
            project=f"{model_name}-regression-tests",
            entity="mosaic-ml",
        )
    ]
    trainer_hparams.set_datadir("~/datasets")
    trainer = trainer_hparams.initialize_object()
    trainer.fit()
    if not is_main_pytest_process:
        return
    metric_monitor = None
    for callback in trainer.state.callbacks:
        if isinstance(callback, MetricMonitor):
            metric_monitor = callback
            break
    assert metric_monitor is not None, "metric monitor should be defined"
    expected_metrics = EXPECTED_METRICS[model_name]
    actual_metrics = metric_monitor.metric_to_latest_value
    errors = []
    for name, (low, high) in expected_metrics.items():
        actual_val = actual_metrics[name]
        if actual_val < low:
            errors.append(f"metric {name} had value {actual_val}, which is below the expected minimum of {low}")
        if actual_val > high:
            errors.append(f"metric {name} had value {actual_val}, which is above the expected maximum of {high}")
    assert len(errors) == 0, f"Metric Errors:\n\n" + "\n".join(errors)
