# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks import OOMObserver
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel, device


@device('cpu', 'gpu')
def test_oom_observer_warnings_on_cpu_models(device: str):
    if version.parse(torch.__version__) <= version.parse('2.1.0.dev'):
        pytest.skip("oom_observer is supported after PyTorch 2.1.0.")

    # Error if the user sets device=cpu even when cuda is available
    del device  # unused. always using cpu
    with pytest.warns(UserWarning, match='The oom observer only works on CUDA devices'):
        Trainer(
            model=SimpleModel(),
            callbacks=OOMObserver(),
            device='cpu',
            train_dataloader=DataLoader(RandomClassificationDataset()),
            max_duration='1ba',
        )


@pytest.mark.gpu
def test_oom_observer():
    if version.parse(torch.__version__) <= version.parse('2.1.0.dev'):
        # memory snapshot is supported after PyTorch 2.1.0.
        return
    # Construct the callbacks
    oom_observer = OOMObserver()

    simple_model = SimpleModel()

    # Construct the trainer and train
    trainer = Trainer(
        model=simple_model,
        callbacks=oom_observer,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='2ba',
    )
    trainer.fit()
