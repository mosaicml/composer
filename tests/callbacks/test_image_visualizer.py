# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import imghdr
import os
import zipfile
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import ImageVisualizer
from composer.loggers import WandBLogger
from composer.loggers.logger import Logger
from composer.trainer import Trainer
from tests.common.datasets import RandomImageDataset
from tests.common.models import SimpleConvModel


@pytest.fixture
def test_wandb_logger(tmp_path, dummy_state):
    pytest.importorskip('wandb', reason='wandb is optional')
    os.environ['WANDB_DIR'] = str(tmp_path)
    os.environ['WANDB_CACHE_DIR'] = str(Path(tmp_path) / Path('wandb_cache'))
    os.environ['WANDB_MODE'] = 'offline'
    dummy_state.run_name = 'wand-test-log-image'
    logger = Logger(dummy_state, [])
    wandb_logger = WandBLogger()
    wandb_logger.init(dummy_state, logger)
    return wandb_logger


@pytest.fixture
def comet_offline_directory(tmp_path):
    return str(tmp_path / Path('my_cometml_runs'))


@pytest.fixture
def comet_logger(monkeypatch, comet_offline_directory):
    comet_ml = pytest.importorskip('comet_ml', reason='comet_ml is optional')

    monkeypatch.setattr(comet_ml, 'Experiment', comet_ml.OfflineExperiment)
    from composer.loggers import CometMLLogger

    # Set offline directory.
    os.environ['COMET_OFFLINE_DIRECTORY'] = comet_offline_directory

    comet_logger = CometMLLogger()
    return comet_logger


def test_image_visualizer_with_wandb(tmp_path, test_wandb_logger):
    wandb = pytest.importorskip('wandb', reason='wandb is optional')

    image_interval = 2
    image_visualizer = ImageVisualizer(interval=f'{image_interval}ba')

    dataset_size = 40
    batch_size = 4
    max_duration = 6
    eval_interval = 6

    trainer = Trainer(model=SimpleConvModel(),
                      callbacks=image_visualizer,
                      loggers=test_wandb_logger,
                      train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_interval=f'{eval_interval}ba',
                      max_duration=f'{max_duration}ba')

    trainer.fit()

    # delete trainer to force WandBLogger to clean up in post_close
    del trainer

    expected_number_train_images = (batch_size * max_duration) / image_interval
    expected_number_eval_images = (max_duration / eval_interval) * batch_size

    # WandB Images are stored in cache when a WandB table is used.
    cache_dir = wandb.env.get_cache_dir()
    all_files_in_cache = []
    for subdir, _, files in os.walk(cache_dir):
        files_in_subdir = [os.path.join(subdir, f) for f in files]
        all_files_in_cache.extend(files_in_subdir)
    imgs = [filepath for filepath in all_files_in_cache if imghdr.what(filepath) == 'png']
    actual_num_images = len(imgs)

    assert actual_num_images == expected_number_eval_images + expected_number_train_images


def test_image_visualizer_with_comet(comet_offline_directory, comet_logger):
    pytest.importorskip('comet_ml', reason='comet_ml is optional')

    image_interval = 2
    image_visualizer = ImageVisualizer(interval=f'{image_interval}ba')

    dataset_size = 40
    batch_size = 4
    max_duration = 6
    eval_interval = 6

    trainer = Trainer(model=SimpleConvModel(),
                      callbacks=image_visualizer,
                      loggers=comet_logger,
                      train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_interval=f'{eval_interval}ba',
                      max_duration=f'{max_duration}ba')

    trainer.fit()

    comet_logger.experiment.end()

    expected_number_train_images = int((batch_size * max_duration) / image_interval)
    expected_number_eval_images = int((max_duration / eval_interval) * batch_size)

    # Extract all files saved to comet offline directory.
    assert comet_logger.experiment is not None
    comet_exp_dump_filepath = str(Path(comet_offline_directory) / Path(comet_logger.experiment.id).with_suffix('.zip'))
    zf = zipfile.ZipFile(comet_exp_dump_filepath)
    zf.extractall(comet_offline_directory)

    # Count the number of files that are images.
    actual_num_images = 0
    for filename in Path(comet_offline_directory).iterdir():
        file_path = str(Path(comet_offline_directory) / Path(filename))
        if imghdr.what(file_path) == 'png':
            actual_num_images += 1
    assert actual_num_images == expected_number_train_images + expected_number_eval_images
