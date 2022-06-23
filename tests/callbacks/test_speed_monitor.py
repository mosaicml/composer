# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import collections.abc
import datetime

from torch.utils.data import DataLoader

from composer.callbacks import SpeedMonitor
from composer.core import Time
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


def _assert_no_negative_values(logged_values):
    for timestamp, loglevel, v in logged_values:
        del timestamp, loglevel  # unused
        if isinstance(v, Time):
            assert int(v) >= 0
        elif isinstance(v, datetime.timedelta):
            assert v.total_seconds() >= 0
        else:
            assert v >= 0


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
        eval_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ep',
    )
    trainer.fit()

    wall_clock_train_calls = len(in_memory_logger.data['wall_clock/train'])
    wall_clock_val_calls = len(in_memory_logger.data['wall_clock/val'])
    wall_clock_total_calls = len(in_memory_logger.data['wall_clock/total'])
    throughput_step_calls = len(in_memory_logger.data['throughput/samples_per_sec'])
    _assert_no_negative_values(in_memory_logger.data['wall_clock/train'])
    _assert_no_negative_values(in_memory_logger.data['wall_clock/val'])
    _assert_no_negative_values(in_memory_logger.data['wall_clock/total'])
    _assert_no_negative_values(in_memory_logger.data['wall_clock/train'])
    _assert_no_negative_values(in_memory_logger.data['throughput/samples_per_sec'])

    assert isinstance(trainer.state.dataloader, collections.abc.Sized)
    assert trainer.state.dataloader_label is not None
    assert trainer.state.dataloader_len is not None
    expected_step_calls = (trainer.state.dataloader_len - speed_monitor.window_size + 1) * int(
        trainer.state.timestamp.epoch)
    assert throughput_step_calls == expected_step_calls
    num_batches = int(trainer.state.timestamp.batch)
    assert wall_clock_total_calls == num_batches
    assert wall_clock_train_calls == num_batches
    assert wall_clock_val_calls == num_batches
