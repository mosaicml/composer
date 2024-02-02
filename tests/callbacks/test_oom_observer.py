# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer import State, Trainer
from composer.callbacks import MemorySnapshot, OOMObserver
from composer.loggers import LoggerDestination
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel, device


@device('cpu', 'gpu')
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.1.0'),
                    reason='OOM Observer requires PyTorch 2.1 or higher')
def test_oom_observer_warnings_on_cpu_models(device: str):

    # Error if the user sets device=cpu even when cuda is available
    del device  # unused. always using cpu
    with pytest.warns(UserWarning, match='OOMObserver only works on CUDA devices, but the model is on cpu.'):
        Trainer(
            model=SimpleModel(),
            callbacks=OOMObserver(),
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
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.1.0'),
                    reason='OOM Observer requires PyTorch 2.1 or higher')
def test_oom_observer():

    # Construct the callbacks
    oom_observer = OOMObserver()

    simple_model = SimpleModel()

    file_tracker_destination = FileUploaderTracker()

    with pytest.raises(torch.cuda.OutOfMemoryError):
        trainer = Trainer(
            model=simple_model,
            loggers=file_tracker_destination,
            callbacks=oom_observer,
            train_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='2ba',
        )

        # trigger OOM
        torch.empty(1024 * 1024 * 1024 * 1024, device='cuda')

        trainer.fit()

    assert len(file_tracker_destination.uploaded_files) == 5


@pytest.mark.gpu
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.1.0'),
                    reason='OOM Observer requires PyTorch 2.1 or higher')
def test_oom_observer_with_memory_snapshot():

    # Construct the callbacks
    oom_observer = OOMObserver()
    memory_snapshot = MemorySnapshot(skip_batches=0, interval='1ba')

    simple_model = SimpleModel()

    file_tracker_destination = FileUploaderTracker()

    trainer = Trainer(
        model=simple_model,
        loggers=file_tracker_destination,
        callbacks=[oom_observer, memory_snapshot],
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='2ba',
    )

    trainer.fit()
    assert len(file_tracker_destination.uploaded_files) == 1
