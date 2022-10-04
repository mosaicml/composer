# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import ImageVisualizer
from composer.core import Time
from composer.loggers import InMemoryLogger, WandBLogger
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

    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)
    num_train_tables = len(in_memory_logger.data['Images/Train'])
    num_eval_tables = len(in_memory_logger.data['Images/Eval'])

    assert isinstance(image_visualizer.interval, Time)
    assert num_train_tables == (num_train_steps - 1) // image_visualizer.interval.value + 1
    assert num_eval_tables == 1


@pytest.mark.skipif(not _WANDB_INSTALLED, reason='Wandb is optional')
def test_wandb_and_image_visualizer(tmp_path):
    import wandb

    image_interval = 2
    image_visualizer = ImageVisualizer(interval=f'{image_interval}ba')
    wandb_logger = WandBLogger(init_kwargs={'dir': tmp_path})

    dataset_size = 40
    batch_size = 4

    trainer = Trainer(model=SimpleConvModel(),
                      callbacks=image_visualizer,
                      loggers=wandb_logger,
                      train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      max_duration='1ep')

    trainer.fit()

    assert wandb.run is not None
    wandb_run_dir = Path(wandb.run.dir)
    run_file = wandb_run_dir.parent / Path(f'run-{wandb.run.id}.wandb')

    # delete trainer to force WandBLogger to clean up in post_close
    del trainer

    # Note, it is not clear how to correctly load this file, so we are just loading it as text
    # and searching the text for expected strings
    with open(run_file, encoding='latin-1') as _wandb_file:
        all_run_text = _wandb_file.read()

    train_metrics_accuracy_count = all_run_text.count('metrics/train/Accuracy')
    eval_metrics_accuracy_count = all_run_text.count('metrics/eval/Accuracy')
    eval_metrics_cross_entropy_count = all_run_text.count('metrics/eval/CrossEntropy')
    train_loss_count = all_run_text.count('loss/train/total')

    expected_number_train_loss_count = (dataset_size / batch_size) + 1  # wandb includes it in the file one extra time
    expected_number_train_metrics_count = (dataset_size /
                                           batch_size) + 2  # wandb includes it in the file two extra times
    expected_number_eval_metrics_count = 2  # wandb includes it in the file twice
    assert train_metrics_accuracy_count == expected_number_train_metrics_count
    assert train_loss_count == expected_number_train_loss_count
    assert eval_metrics_accuracy_count == expected_number_eval_metrics_count
    assert eval_metrics_cross_entropy_count == expected_number_eval_metrics_count

    wandb_media_dir = wandb_run_dir.parent / Path('files') / Path('media') / Path('table') / Path('Images')
    image_files = list(os.listdir(wandb_media_dir))
    train_image_files_count = sum([1 for file in image_files if file.startswith('Train')])
    eval_image_files_count = sum([1 for file in image_files if file.startswith('Eval')])

    expected_number_train_images = (dataset_size / batch_size) / image_interval
    expected_number_eval_images = 1
    assert train_image_files_count == expected_number_train_images
    assert eval_image_files_count == expected_number_eval_images
