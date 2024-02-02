# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer import State, Trainer
from composer.callbacks import MemorySnapshot
from composer.loggers import LoggerDestination
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.1.0'),
                    reason='OOM Observer requires PyTorch 2.1 or higher')
def test_memory_snapshot_warnings_on_cpu_models():
    with pytest.warns(UserWarning):
        Trainer(
            model=SimpleModel(),
            callbacks=MemorySnapshot(),
            device='cpu',
            train_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='1ba',
        )


class FileUploaderTracker(LoggerDestination):

    def __init__(self) -> None:
        self.uploaded_files = []

    def upload_file(self, state: State, remote_file_name: str, file_path: pathlib.Path, *, overwrite: bool):
        del state, overwrite  # unused
        self.uploaded_files.append((remote_file_name, file_path))


@pytest.mark.gpu
@pytest.mark.parametrize('interval', ['1ba'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.1.0'),
                    reason='OOM Observer requires PyTorch 2.1 or higher')
def test_memory_snapshot(interval: str):
    # Construct the callbacks
    skip_batches = 0
    memory_snapshot = MemorySnapshot(skip_batches=skip_batches, interval=interval)
    simple_model = SimpleModel()
    file_tracker_destination = FileUploaderTracker()

    # Construct the trainer and train
    trainer = Trainer(
        model=simple_model,
        loggers=file_tracker_destination,
        callbacks=memory_snapshot,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='2ba',
    )
    trainer.fit()
    assert len(file_tracker_destination.uploaded_files) == 1
    trainer.close()
