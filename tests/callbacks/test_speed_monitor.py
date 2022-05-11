# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import collections.abc
from typing import cast
from unittest.mock import MagicMock

from composer.callbacks import SpeedMonitorHparams
from composer.loggers.logger_destination import LoggerDestination
from composer.trainer import TrainerHparams
from composer.utils import ensure_tuple


def test_speed_monitor(composer_trainer_hparams: TrainerHparams):
    speed_monitor_hparams = SpeedMonitorHparams(window_size=2)
    composer_trainer_hparams.callbacks.append(speed_monitor_hparams)
    composer_trainer_hparams.grad_accum = 1
    composer_trainer_hparams.train_batch_size = 10
    max_epochs = 2
    composer_trainer_hparams.max_duration = f"{max_epochs}ep"
    trainer = composer_trainer_hparams.initialize_object()
    log_destination = MagicMock()
    log_destination = cast(LoggerDestination, log_destination)
    trainer.logger.destinations = ensure_tuple(log_destination)
    trainer.fit()

    throughput_epoch_calls = 0
    wall_clock_train_calls = 0
    throughput_step_calls = 0
    for call_ in log_destination.log_data.mock_calls:
        metrics = call_[1][2]
        if "throughput/step" in metrics:
            throughput_step_calls += 1
        if "throughput/epoch" in metrics:
            throughput_epoch_calls += 1
        if 'wall_clock_train' in metrics:
            wall_clock_train_calls += 1

    assert isinstance(trainer.state.dataloader, collections.abc.Sized)
    assert trainer.state.dataloader_label is not None
    assert trainer.state.dataloader_len is not None
    expected_step_calls = (trainer.state.dataloader_len - speed_monitor_hparams.window_size) * max_epochs
    assert throughput_step_calls == expected_step_calls
    assert throughput_epoch_calls == max_epochs
    assert wall_clock_train_calls == max_epochs
