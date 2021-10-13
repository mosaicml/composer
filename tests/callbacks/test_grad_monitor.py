# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import MagicMock

import pytest

from composer.callbacks import GradMonitorHparams
from composer.datasets.synthetic import SyntheticDatasetHparams
from composer.trainer import TrainerHparams


def _do_trainer_fit(mosaic_trainer_hparams: TrainerHparams, log_layers=False):
    grad_monitor_hparams = GradMonitorHparams(log_layer_grad_norms=log_layers)
    mosaic_trainer_hparams.callbacks.append(grad_monitor_hparams)
    mosaic_trainer_hparams.ddp.fork_rank_0 = False
    mosaic_trainer_hparams.max_epochs = 1

    mosaic_trainer_hparams.total_batch_size = 50
    trainer = mosaic_trainer_hparams.initialize_object()
    log_destination = MagicMock()
    log_destination.will_log.return_value = True
    trainer.logger.backends = [log_destination]
    trainer.fit()

    assert isinstance(mosaic_trainer_hparams.train_dataset, SyntheticDatasetHparams)
    num_train_samples = mosaic_trainer_hparams.train_dataset.sample_pool_size
    num_train_steps = num_train_samples // mosaic_trainer_hparams.total_batch_size

    return log_destination, num_train_steps


@pytest.mark.timeout(60)
@pytest.mark.run_long
def test_grad_monitor_no_layers(mosaic_trainer_hparams: TrainerHparams):
    log_destination, num_train_steps = _do_trainer_fit(mosaic_trainer_hparams, log_layers=False)
    grad_norm_calls = 0
    for log_call in log_destination.log_metric.mock_calls:
        metrics = log_call[1][3]
        if "grad_l2_norm/step" in metrics:
            grad_norm_calls += 1

    # expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps


@pytest.mark.timeout(60)
@pytest.mark.run_long
def test_grad_moniter_per_layer(mosaic_trainer_hparams: TrainerHparams):
    log_destination, num_train_steps = _do_trainer_fit(mosaic_trainer_hparams, log_layers=True)
    layer_norm_calls = 0
    for log_call in log_destination.log_metric.mock_calls:
        metrics = log_call[1][3]
        if not isinstance(metrics, dict):
            continue
        if any(map(lambda x: "layer_grad_l2_norm" in x, metrics.keys())):
            layer_norm_calls += 1

    # expected to log layer grad norms once per training step
    assert layer_norm_calls == num_train_steps
