# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import imghdr
import os
import zipfile
from collections import defaultdict
from json import JSONDecoder
from pathlib import Path
from typing import Sequence

import pytest
import torch
from torch.utils.data import DataLoader

from composer.loggers import CometMLLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


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


def test_comet_ml_log_image_saves_images(comet_logger: CometMLLogger, comet_offline_directory: str):
    assert isinstance(comet_offline_directory, str)

    # We group all the image size variants into one test because calling comet_experiment.end() is slow
    image_variants = [
        (torch.rand(4), False),  # 1D image
        (torch.rand(4, 4), False),  # 2D image
        (torch.rand(4, 4, 3), True),  # with channels, channels last
        (torch.rand(3, 4, 4), False),  # with channels, not channels last
        (torch.rand(2, 4, 4, 3), True),  # multiple images in tensor
        ([torch.rand(4, 4, 3), torch.rand(4, 4, 3)], True)  # multiple images in list
    ]

    expected_num_images_total = 0
    for (images, channels_last) in image_variants:
        # Count expected images and generate numpy arrays from torch tensors.
        if isinstance(images, Sequence):
            expected_num_images = len(images)
            np_images = [image.numpy() for image in images]

        else:
            expected_num_images = 1 if images.ndim < 4 else images.shape[0]
            np_images = images.numpy()

        # Log images from torch tensors and numpy arrays.
        comet_logger.log_images(images, channels_last=channels_last)
        comet_logger.log_images(np_images, channels_last=channels_last)
        expected_num_images *= 2  # One set of torch tensors, one set of numpy arrays
        expected_num_images_total += expected_num_images

    comet_logger.post_close()

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
    assert actual_num_images == expected_num_images_total


def test_comet_ml_log_image_saves_images_with_masks(comet_logger: CometMLLogger, comet_offline_directory: str):
    assert isinstance(comet_offline_directory, str)

    # We group all the image size variants into one test because calling comet_experiment.end() is slow
    image_variants = [
        # channels last
        # single image, single mask
        (torch.rand(4, 4, 3), {
            'pred': torch.randint(0, 10, (4, 4))
        }, True),
        # multiple images, masks in tensor
        (torch.rand(2, 4, 4, 3), {
            'pred': torch.randint(0, 10, (2, 4, 4))
        }, True),
        # multiple images, masks in last
        (torch.rand(2, 4, 4, 3), {
            'pred': 2 * [torch.randint(0, 10, (4, 4))]
        }, True),
        # multiple images, multiple masks
        (torch.rand(2, 4, 4, 3), {
            'pred': torch.randint(0, 10, (2, 4, 4)),
            'pred2': torch.randint(0, 10, (2, 4, 4))
        }, True),

        # not channels last
        # single image, single mask
        (torch.rand(3, 4, 4), {
            'pred': torch.randint(0, 10, (4, 4))
        }, False),
        # multiple images, masks in tensor
        (torch.rand(2, 3, 4, 4), {
            'pred': torch.randint(0, 10, (2, 4, 4))
        }, False)
    ]

    expected_num_masks_and_images_total = 0
    for (images, masks, channels_last) in image_variants:
        # Count expected images and generate numpy arrays from torch tensors.
        num_masks = len(masks.keys())
        num_images = images.shape[0] if images.ndim == 4 else 1
        num_additional_images_per_mask = 2  # Mask overlaid on image + mask by itself.
        expected_num_masks = num_images * num_additional_images_per_mask * num_masks
        expected_num_masks_and_images = num_images + expected_num_masks
        expected_num_masks_and_images_total += expected_num_masks_and_images

        # Log images from torch tensors and numpy arrays.
        comet_logger.log_images(images, masks=masks, channels_last=channels_last)

    comet_logger.post_close()

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
    assert actual_num_images == expected_num_masks_and_images_total


def test_comet_ml_logging_train_loop(monkeypatch, tmp_path):
    comet_ml = pytest.importorskip('comet_ml', reason='comet_ml is optional')

    monkeypatch.setattr(comet_ml, 'Experiment', comet_ml.OfflineExperiment)
    from composer.loggers import CometMLLogger

    # Set offline directory.
    offline_directory = str(tmp_path / Path('.my_cometml_runs'))
    os.environ['COMET_OFFLINE_DIRECTORY'] = offline_directory

    comet_logger = CometMLLogger()

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=2,
        max_duration='2ep',
        loggers=comet_logger,
    )
    trainer.fit()

    run_name = trainer.state.run_name

    del trainer

    assert comet_logger.experiment is not None
    assert comet_logger.experiment.ended

    # Open, decompress, decode, and extract offline dump of metrics.
    comet_exp_dump_filepath = str(Path(offline_directory) / Path(comet_logger.experiment.id).with_suffix('.zip'))
    zf = zipfile.ZipFile(comet_exp_dump_filepath)
    comet_logs_path = zf.extract('messages.json', path=offline_directory)
    jd = JSONDecoder()
    msg_type_to_msgs = defaultdict(list)

    with open(comet_logs_path) as f:
        for line in f.readlines():
            parsed_line = jd.decode(line)
            msg_type_to_msgs[parsed_line['type']].append(parsed_line['payload'])

    # Check that init set the run name
    assert comet_logger.name == run_name
    assert comet_logger.experiment.name == run_name

    # Check that basic metrics appear in the comet logs
    assert len([
        metric_msg for metric_msg in msg_type_to_msgs['metric_msg']
        if metric_msg['metric']['metricName'] == 'trainer/epoch'
    ]) == 2

    # Check that basic params appear in the comet logs
    assert len([
        param_msg for param_msg in msg_type_to_msgs['parameter_msg']
        if param_msg['param']['paramName'] == 'rank_zero_seed'
    ]) > 0


def test_comet_ml_log_metrics_and_hyperparameters(monkeypatch, tmp_path):
    """Check metrics logged with CometMLLogger are properly written to offline dump."""
    pytest.importorskip('comet_ml', reason='comet_ml is optional')
    import comet_ml

    # Set some dummy log values.
    steps = [0, 1, 2]
    metric_values = [0.1, 0.4, 0.7]
    metric_name = 'my_test_metric'
    param_names = ['my_cool_parameter1', 'my_cool_parameter2']
    param_values = [10, 3]

    # Set offline directory.
    offline_directory = str(tmp_path / Path('.my_cometml_runs'))
    os.environ['COMET_OFFLINE_DIRECTORY'] = offline_directory

    # Monkeypatch Experiment with OfflineExperiment to avoid uploading to CometML and
    # avoid needing an API+KEY.
    monkeypatch.setattr(comet_ml, 'Experiment', comet_ml.OfflineExperiment)
    from composer.loggers import CometMLLogger

    # Log dummy values with CometMLLogger.
    comet_logger = CometMLLogger()
    comet_logger.log_hyperparameters(dict(zip(param_names, param_values)))
    for step, metric_value in zip(steps, metric_values):
        comet_logger.log_metrics({'my_test_metric': metric_value}, step=step)

    # Simulate the post_close call to end the CometML experiment
    comet_logger.post_close()

    assert comet_logger.experiment is not None
    # Check that calling post_close ended the comet experiment
    assert comet_logger.experiment.ended

    # Open, decompress, decode, and extract offline dump of metrics.
    comet_exp_dump_filepath = str(Path(offline_directory) / Path(comet_logger.experiment.id).with_suffix('.zip'))
    zf = zipfile.ZipFile(comet_exp_dump_filepath)
    comet_logs_path = zf.extract('messages.json', path=offline_directory)
    jd = JSONDecoder()
    created_from_found = False
    expected_created_from_log = {'key': 'Created from', 'val': 'mosaicml-composer'}
    metric_msgs = []
    param_msgs = []
    with open(comet_logs_path) as f:
        for line in f.readlines():
            comet_msg = jd.decode(line)
            if comet_msg['type'] == 'ws_msg' and comet_msg['payload'].get('log_other', {}) == expected_created_from_log:
                created_from_found = True
            if (comet_msg['type'] == 'metric_msg') and (comet_msg['payload']['metric']['metricName']
                                                        == 'my_test_metric'):
                metric_msgs.append(comet_msg['payload']['metric'])
            if comet_msg['type'] == 'parameter_msg' and (
                    comet_msg['payload']['param']['paramName'].startswith('my_cool')):
                param_msgs.append(comet_msg['payload']['param'])

    # Check that the "Created from key was properly set"
    assert created_from_found

    # Assert dummy metrics input to log_metrics are the same as
    # those written to offline dump.
    assert [msg['metricValue'] for msg in metric_msgs] == metric_values
    assert [msg['step'] for msg in metric_msgs] == steps
    assert all([msg['metricName'] == metric_name for msg in metric_msgs])

    # Assert dummy params input to log_hyperparameters are the same as
    # those written to offline dump
    assert [msg['paramValue'] for msg in param_msgs] == param_values
    assert [msg['paramName'] for msg in param_msgs] == param_names
