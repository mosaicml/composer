# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
from torch.utils.data import DataLoader

from composer import State, Trainer
from composer.callbacks import MemorySnapshot
from composer.loggers import LoggerDestination
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


class FileUploaderTracker(LoggerDestination):

    def __init__(self) -> None:
        self.uploaded_files = []

    def upload_file(self, state: State, remote_file_name: str, file_path: pathlib.Path, *, overwrite: bool):
        del state, overwrite  # unused
        self.uploaded_files.append((remote_file_name, file_path))


@pytest.mark.parametrize('interval', ['1ba', '3ba'])
def test_memory_snapshot(interval: str, tmp_path: pathlib.Path):
    # Construct the callbacks
    skip_batches = 1
    memory_snapshot = MemorySnapshot(skip_batches=skip_batches, interval=interval)

    simple_model = SimpleModel()

    file_tracker_destination = FileUploaderTracker()

    # Construct the trainer and train
    trainer = Trainer(
        model=simple_model,
        loggers=[file_tracker_destination],
        callbacks=memory_snapshot,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='10ba',
    )
    trainer.fit()
    assert len(file_tracker_destination.uploaded_files) == 1
