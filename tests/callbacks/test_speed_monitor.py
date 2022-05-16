# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import collections.abc

from torch.utils.data import DataLoader

from composer.callbacks import SpeedMonitor
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


def test_speed_monitor():
    # Construct the callbacks
    speed_monitor = SpeedMonitor(window_size=2)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger

    # Construct the trainer and train
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=speed_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration="1ep",
    )
    trainer.fit()

    throughput_epoch_calls = 0
    wall_clock_train_calls = 0
    throughput_step_calls = 0
    for timestamp, log_level, metrics in in_memory_logger.data:
        del timestamp, log_level  # unused
        if "throughput/step" in metrics:
            throughput_step_calls += 1
        if "throughput/epoch" in metrics:
            throughput_epoch_calls += 1
        if 'wall_clock_train' in metrics:
            wall_clock_train_calls += 1

    assert isinstance(trainer.state.dataloader, collections.abc.Sized)
    assert trainer.state.dataloader_label is not None
    assert trainer.state.dataloader_len is not None
    expected_step_calls = (trainer.state.dataloader_len - speed_monitor.window_size) * int(
        trainer.state.timestamp.epoch)
    assert throughput_step_calls == expected_step_calls
    assert throughput_epoch_calls == int(trainer.state.timestamp.epoch)
    assert wall_clock_train_calls == int(trainer.state.timestamp.epoch)
