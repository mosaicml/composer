# Copyright 2021 MosaicML. All Rights Reserved.

import collections.abc
from unittest.mock import MagicMock

import pytest

from composer.callbacks import SpeedMonitorHparams
from composer.trainer import TrainerHparams


@pytest.mark.timeout(60)
@pytest.mark.run_long
def test_speed_monitor(mosaic_trainer_hparams: TrainerHparams):
    speed_monitor_hparams = SpeedMonitorHparams(window_size=2)
    mosaic_trainer_hparams.callbacks.append(speed_monitor_hparams)
    mosaic_trainer_hparams.grad_accum = 1
    mosaic_trainer_hparams.ddp.fork_rank_0 = False
    mosaic_trainer_hparams.total_batch_size = 10
    mosaic_trainer_hparams.max_epochs = 2
    trainer = mosaic_trainer_hparams.initialize_object()
    log_destination = MagicMock()
    log_destination.will_log.return_value = True
    trainer.logger.backends = [log_destination]
    trainer.fit()

    throughput_epoch_calls = 0
    wall_clock_train_calls = 0
    throughput_step_calls = 0
    for call_ in log_destination.log_metric.mock_calls:
        metrics = call_[1][3]
        if "throughput/step" in metrics:
            throughput_step_calls += 1
        if "throughput/epoch" in metrics:
            throughput_epoch_calls += 1
        if 'wall_clock_train' in metrics:
            wall_clock_train_calls += 1

    assert isinstance(trainer.state.train_dataloader, collections.abc.Sized)
    expected_step_calls = (len(trainer.state.train_dataloader) -
                           speed_monitor_hparams.window_size) * mosaic_trainer_hparams.max_epochs
    assert throughput_step_calls == expected_step_calls
    assert throughput_epoch_calls == mosaic_trainer_hparams.max_epochs
    assert wall_clock_train_calls == mosaic_trainer_hparams.max_epochs
