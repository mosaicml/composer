# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import imghdr
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import ImageVisualizer
from composer.loggers import WandBLogger
from composer.loggers.logger import Logger
from composer.trainer import Trainer
from tests.common.datasets import RandomImageDataset, RandomSegmentationDataset
from tests.common.models import SimpleConvModel, SimpleSegmentationModel


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


def test_image_visualizer_segmentation_with_wandb(test_wandb_logger, monkeypatch):
    pytest.importorskip('wandb', reason='wandb is optional')

    image_interval = 2
    image_visualizer = ImageVisualizer(interval=f'{image_interval}ba', mode='segmentation')

    dataset_size = 40
    batch_size = 2
    max_duration = 4
    eval_interval = 4
    num_classes = 2
    num_channels = 3

    log_image_mock = MagicMock()
    monkeypatch.setattr(test_wandb_logger, 'log_images', log_image_mock)
    trainer = Trainer(model=SimpleSegmentationModel(num_channels=num_channels, num_classes=num_classes),
                      callbacks=image_visualizer,
                      loggers=test_wandb_logger,
                      train_dataloader=DataLoader(
                          RandomSegmentationDataset(size=dataset_size, shape=(3, 8, 8), num_classes=num_classes),
                          batch_size),
                      eval_dataloader=DataLoader(
                          RandomSegmentationDataset(size=dataset_size, shape=(3, 8, 8), num_classes=num_classes),
                          batch_size),
                      eval_interval=f'{eval_interval}ba',
                      max_duration=f'{max_duration}ba')

    trainer.fit()

    # delete trainer to force WandBLogger to clean up in post_close
    del trainer

    # Test is flaky if we check cache dir for image files, so we count the log calls instead.
    expected_num_log_image_calls_train = int(max_duration / image_interval)
    expected_num_log_image_calls_eval = int(max_duration / eval_interval)

    assert log_image_mock.call_count == expected_num_log_image_calls_train + expected_num_log_image_calls_eval


def test_image_visualizer_segmentation_with_comet(comet_offline_directory, comet_logger):
    pytest.importorskip('comet_ml', reason='comet_ml is optional')

    image_interval = 2
    image_visualizer = ImageVisualizer(interval=f'{image_interval}ba', mode='segmentation')

    dataset_size = 40
    batch_size = 4
    max_duration = 6
    eval_interval = 6
    num_classes = 2
    num_channels = 3
    num_masks = 2

    trainer = Trainer(model=SimpleSegmentationModel(num_channels=num_channels, num_classes=num_classes),
                      callbacks=image_visualizer,
                      loggers=comet_logger,
                      train_dataloader=DataLoader(
                          RandomSegmentationDataset(size=dataset_size, shape=(3, 32, 32), num_classes=num_classes),
                          batch_size),
                      eval_dataloader=DataLoader(
                          RandomSegmentationDataset(size=dataset_size, shape=(3, 32, 32), num_classes=num_classes),
                          batch_size),
                      eval_interval=f'{eval_interval}ba',
                      max_duration=f'{max_duration}ba')

    trainer.fit()

    # delete trainer to force WandBLogger to clean up in post_close
    comet_logger.experiment.end()

    expected_number_train_images = (batch_size * max_duration) / image_interval
    expected_number_eval_images = (max_duration / eval_interval) * batch_size
    num_additional_images_per_mask = 2  # Mask overlaid on image + mask by itself.
    expected_num_masks = num_masks * num_additional_images_per_mask * (expected_number_train_images +
                                                                       expected_number_eval_images)

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
    assert actual_num_images == expected_number_train_images + expected_number_eval_images + expected_num_masks
