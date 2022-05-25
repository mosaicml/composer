# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from unittest.mock import MagicMock

from composer.callbacks import GradMonitorHparams
from composer.loggers import LoggerDestination
from composer.trainer import TrainerHparams
from composer.utils import ensure_tuple


def _do_trainer_fit(composer_trainer_hparams: TrainerHparams, log_layers: bool = False):
    grad_monitor_hparams = GradMonitorHparams(log_layer_grad_norms=log_layers)
    composer_trainer_hparams.callbacks.append(grad_monitor_hparams)
    composer_trainer_hparams.max_duration = "1ep"

    trainer = composer_trainer_hparams.initialize_object()
    log_destination = MagicMock()
    log_destination = cast(LoggerDestination, log_destination)
    trainer.logger.destinations = ensure_tuple(log_destination)
    trainer.fit()

    num_train_steps = composer_trainer_hparams.train_subset_num_batches

    return log_destination, num_train_steps


def test_grad_monitor_no_layers(composer_trainer_hparams: TrainerHparams):
    log_destination, num_train_steps = _do_trainer_fit(composer_trainer_hparams, log_layers=False)
    grad_norm_calls = 0
    for log_call in log_destination.log_data.mock_calls:
        metrics = log_call[1][2]
        if "grad_l2_norm/step" in metrics:
            grad_norm_calls += 1

    # expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps


def test_grad_monitor_per_layer(composer_trainer_hparams: TrainerHparams):
    log_destination, num_train_steps = _do_trainer_fit(composer_trainer_hparams, log_layers=True)
    layer_norm_calls = 0
    for log_call in log_destination.log_data.mock_calls:
        metrics = log_call[1][2]
        if not isinstance(metrics, dict):
            continue
        if any(map(lambda x: "layer_grad_l2_norm" in x, metrics.keys())):
            layer_norm_calls += 1

    # expected to log layer grad norms once per training step
    assert layer_norm_calls == num_train_steps
