# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks import MemorySnapshot
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel, device


@device('cpu', 'gpu')
def test_memory_snapshot_warnings_on_cpu_models(device: str):
    if version.parse(torch.__version__) <= version.parse('2.1.0.dev'):
        # memory snapshot is supported after PyTorch 2.1.0.
        return
    # Error if the user sets device=cpu even when cuda is available
    del device  # unused. always using cpu
    with pytest.warns(UserWarning, match='The memory snapshot only works on CUDA devices'):
        Trainer(
            model=SimpleModel(),
            callbacks=MemorySnapshot(),
            device='cpu',
            train_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='1ba',
        )


@pytest.mark.gpu
@pytest.mark.parametrize('interval', ['1ba'])
def test_memory_snapshot(interval: str):
    if version.parse(torch.__version__) <= version.parse('2.1.0.dev'):
        # memory snapshot is supported after PyTorch 2.1.0.
        return
    # Construct the callbacks
    skip_batches = 0
    memory_snapshot = MemorySnapshot(skip_batches=skip_batches, interval=interval)

    simple_model = SimpleModel()

    # Construct the trainer and train
    trainer = Trainer(
        model=simple_model,
        callbacks=memory_snapshot,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ba',
    )
    trainer.fit()
    trainer.close()
