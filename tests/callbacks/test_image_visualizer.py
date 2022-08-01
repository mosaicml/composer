# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import ImageVisualizer
from composer.core import Time
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common.datasets import RandomImageDataset
from tests.common.models import SimpleConvModel

try:
    import wandb
    _WANDB_INSTALLED = True
    del wandb  # unused
except ImportError:
    _WANDB_INSTALLED = False


@pytest.mark.skipif(not _WANDB_INSTALLED, reason='Wandb is optional')
@pytest.mark.parametrize('interval', ['9ba', '90ba'])
def test_image_visualizer(interval: str):
    # Construct the callback
    image_visualizer = ImageVisualizer(interval=interval)
    in_memory_logger = InMemoryLogger()  # track the logged images in the in_memory_logger

    # Construct the trainer and train
    trainer = Trainer(
        model=SimpleConvModel(),
        callbacks=image_visualizer,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomImageDataset()),
        eval_dataloader=DataLoader(RandomImageDataset()),
        max_duration='1ep',
    )
    pytest.xfail('This test segfaults. See https://mosaicml.atlassian.net/browse/CO-776')
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)
    num_train_tables = len(in_memory_logger.data['Images/Train'])
    num_eval_tables = len(in_memory_logger.data['Images/Eval'])

    assert isinstance(image_visualizer.interval, Time)
    assert num_train_tables == (num_train_steps - 1) // image_visualizer.interval.value + 1
    assert num_eval_tables == 1
