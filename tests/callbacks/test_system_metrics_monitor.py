# Copyright 2022-2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import SystemMetricsMonitor
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.gpu
def test_system_metrics_monitor_gpu():
    # Construct the trainer
    system_metrics_monitor = SystemMetricsMonitor(gpu_available=True)
    in_memory_logger = InMemoryLogger()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=system_metrics_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ba',
    )
    trainer.fit()

    assert 'device0_gpu_percentage' in in_memory_logger.data
    assert 'cpu_percentage' in in_memory_logger.data


def test_system_metrics_monitor_cpu():
    # Construct the trainer
    system_metrics_monitor = SystemMetricsMonitor()
    in_memory_logger = InMemoryLogger()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=system_metrics_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ba',
    )
    trainer.fit()

    assert 'cpu_percentage' in in_memory_logger.data
