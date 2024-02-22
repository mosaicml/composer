# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from composer.utils import dist, get_device
from tests.common import RandomClassificationDataset, SimpleModel


def test_memory_monitor_warnings_on_cpu_models():
    with pytest.warns(UserWarning, match='The memory monitor only works on CUDA devices'):
        Trainer(
            model=SimpleModel(),
            callbacks=MemoryMonitor(),
            device='cpu',
            train_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='1ba',
        )


@pytest.mark.gpu
def test_memory_monitor_gpu():
    # Construct the trainer
    memory_monitor = MemoryMonitor()
    in_memory_logger = InMemoryLogger()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=memory_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ba',
    )
    trainer.fit()

    num_memory_monitor_calls = len(in_memory_logger.data['memory/peak_allocated_mem'])

    assert num_memory_monitor_calls == int(trainer.state.timestamp.batch)


@pytest.mark.gpu
@pytest.mark.world_size(2)
def test_dist_memory_monitor_gpu():
    dist.initialize_dist(get_device(None))

    # Construct the trainer
    memory_monitor = MemoryMonitor(dist_aggregate_batch_interval=1)
    in_memory_logger = InMemoryLogger()

    # Add extra memory useage to rank 1
    numel = 1 << 30  # about 1B elements in 32 bits is about 4GB
    expected_extra_mem_usage_gb = 4 * numel / 1e9
    if dist.get_local_rank() == 1:
        _ = torch.randn(numel, device='cuda')

    dataset = RandomClassificationDataset()
    trainer = Trainer(
        model=SimpleModel(),
        callbacks=memory_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(dataset=dataset, sampler=DistributedSampler(dataset=dataset)),
        max_duration='2ba',
    )
    trainer.fit()

    peak_allocated_mem = in_memory_logger.data['memory/peak_allocated_mem'][-1][-1]
    peak_allocated_mem_max = in_memory_logger.data['memory/peak_allocated_mem_max'][-1][-1]

    gb_buffer = 0.5
    extra_mem_gb = expected_extra_mem_usage_gb - gb_buffer
    if dist.get_local_rank() == 0:
        assert peak_allocated_mem_max - extra_mem_gb >= peak_allocated_mem
