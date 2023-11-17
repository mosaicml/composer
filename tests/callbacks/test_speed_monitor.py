# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import collections.abc
import datetime

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import SpeedMonitor
from composer.core import Time
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


def _assert_no_negative_values(logged_values):
    for timestamp, v in logged_values:
        del timestamp  # unused
        if isinstance(v, Time):
            assert int(v) >= 0
        elif isinstance(v, datetime.timedelta):
            assert v.total_seconds() >= 0
        else:
            assert v >= 0


@pytest.mark.parametrize('flops_per_batch', [False, True])
def test_speed_monitor(flops_per_batch: bool):
    # Construct the callbacks
    speed_monitor = SpeedMonitor(window_size=2)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger

    model = SimpleModel()
    if flops_per_batch:
        model.flops_per_batch = lambda batch: len(batch) * 100.0

    # Construct the trainer and train
    trainer = Trainer(
        model=model,
        callbacks=speed_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        eval_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ep',
    )
    trainer.fit()

    _assert_no_negative_values(in_memory_logger.data['time/train'])
    _assert_no_negative_values(in_memory_logger.data['time/val'])
    _assert_no_negative_values(in_memory_logger.data['time/total'])
    _assert_no_negative_values(in_memory_logger.data['throughput/batches_per_sec'])
    _assert_no_negative_values(in_memory_logger.data['throughput/samples_per_sec'])
    _assert_no_negative_values(in_memory_logger.data['throughput/device/batches_per_sec'])
    _assert_no_negative_values(in_memory_logger.data['throughput/device/samples_per_sec'])
    if flops_per_batch:
        _assert_no_negative_values(in_memory_logger.data['throughput/flops_per_sec'])
        _assert_no_negative_values(in_memory_logger.data['throughput/device/flops_per_sec'])

    assert isinstance(trainer.state.dataloader, collections.abc.Sized)
    assert trainer.state.dataloader_label is not None
    assert trainer.state.dataloader_len is not None
    expected_step_calls = (trainer.state.dataloader_len - len(speed_monitor.history_samples) + 1) * int(
        trainer.state.timestamp.epoch)
    assert len(in_memory_logger.data['throughput/batches_per_sec']) == expected_step_calls
    assert len(in_memory_logger.data['throughput/samples_per_sec']) == expected_step_calls
    assert len(in_memory_logger.data['throughput/device/batches_per_sec']) == expected_step_calls
    assert len(in_memory_logger.data['throughput/device/samples_per_sec']) == expected_step_calls
    if flops_per_batch:
        assert len(in_memory_logger.data['throughput/flops_per_sec']) == expected_step_calls
        assert len(in_memory_logger.data['throughput/device/flops_per_sec']) == expected_step_calls
    num_batches = int(trainer.state.timestamp.batch)
    assert len(in_memory_logger.data['time/total']) == num_batches
    assert len(in_memory_logger.data['time/train']) == num_batches
    assert len(in_memory_logger.data['time/val']) == num_batches
