# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
import uuid
from pathlib import Path
from typing import Generator, Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from composer import Trainer
from composer._version import __version__
from composer.loggers import NeptuneLogger
from composer.utils import dist
from tests.common import RandomImageDataset, SimpleConvModel
from tests.common.markers import device


@pytest.fixture
def test_neptune_logger() -> NeptuneLogger:
    neptune_project = 'test_project'
    neptune_api_token = 'test_token'

    neptune_logger = NeptuneLogger(
        project=neptune_project,
        api_token=neptune_api_token,
        rank_zero_only=False,
        mode='debug',
        upload_checkpoints=True,
    )

    return neptune_logger


def test_neptune_init(test_neptune_logger):
    mock_state = MagicMock()
    mock_state.run_name = 'dummy-run-name'  # should appear in sys/tags

    test_neptune_logger.init(state=mock_state, logger=MagicMock())

    assert test_neptune_logger.neptune_run is not None

    test_neptune_logger.neptune_run.sync()
    assert test_neptune_logger.neptune_run[NeptuneLogger.integration_version_key].fetch() == __version__
    assert test_neptune_logger.neptune_run['sys/name'].fetch() == 'dummy-run-name'
    assert test_neptune_logger.base_handler['rank'].fetch() == 0


@device('cpu')
def test_neptune_logging(device, test_neptune_logger):

    dataset_size = 64
    batch_size = 4
    num_batches = 4
    eval_interval = '1ba'

    trainer = Trainer(
        model=SimpleConvModel(),
        loggers=test_neptune_logger,
        train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
        eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
        max_duration=f'{num_batches}ba',
        eval_interval=eval_interval,
        device=device,
    )
    trainer.fit()

    assert test_neptune_logger.neptune_run is not None
    assert test_neptune_logger.base_handler is not None

    for metric_name in [
        'metrics/train/MulticlassAccuracy',
        'metrics/eval/MulticlassAccuracy',
        'metrics/eval/CrossEntropy',
        'loss/train/total',
    ]:
        path = f'{test_neptune_logger._base_namespace}/{test_neptune_logger.metric_namespace}/{metric_name}'
        assert test_neptune_logger.neptune_run.exists(path)

    for hyperparam_name in ['node_name', 'num_cpus_per_node', 'num_nodes', 'rank_zero_seed']:
        path = f'{test_neptune_logger._base_namespace}/{test_neptune_logger.hyperparam_namespace}/{hyperparam_name}'
        assert test_neptune_logger.neptune_run.exists(path)

    assert test_neptune_logger.base_handler['hyperparameters/num_nodes'].fetch() == 1


@pytest.mark.gpu
@pytest.mark.world_size(1, 2)
def test_upload_and_download_file(test_neptune_logger, tmp_path, dummy_state):
    neptune_artifact_name = 'test-neptune-artifact-' + str(uuid.uuid4())
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = Path(tmp_paths[0])
    file_content = 'hello from Neptune!'

    dummy_neptune_artifact_path = save_folder / 'neptune_artifact.txt'
    if dist.get_global_rank() == 0:
        with open(dummy_neptune_artifact_path, 'w+') as f:
            f.write(file_content)

    test_neptune_logger.upload_file(
        state=dummy_state,
        file_path=dummy_neptune_artifact_path,
        remote_file_name=neptune_artifact_name,
    )

    dist.barrier()

    assert test_neptune_logger.neptune_run.exists(f'{test_neptune_logger._base_namespace}/{neptune_artifact_name}')

    dst_path = save_folder / 'neptune_artifact'

    test_neptune_logger.download_file(
        remote_file_name=neptune_artifact_name,
        destination=str(dst_path),
    )

    assert dst_path.exists()

    with open(str(dst_path), 'r') as fp:
        assert fp.read() == file_content


def test_neptune_log_image(test_neptune_logger):
    pytest.importorskip('neptune', reason='neptune is optional')

    with patch('neptune.attributes.FileSeries.extend', MagicMock()) as mock_extend:
        image_variants = [
            (torch.rand(4, 4), False),  # 2D image
            (torch.rand(2, 3, 4, 4), False),  # multiple images, not channels last
            (torch.rand(2, 3, 4, 4, dtype=torch.float64), False),  # same as above but with float64
            (torch.rand(3, 4, 4), False),  # with channels, not channels last
            ([torch.rand(4, 4, 3)], True),  # with channels, channels last
            (torch.rand(2, 4, 4, 3), True),  # multiple images, channels last
            ([torch.rand(4, 4, 3), torch.rand(4, 4, 3)], True),  # multiple images in list
        ]

        expected_num_images_total = 0
        for (images, channels_last) in image_variants:
            if isinstance(images, Sequence):
                expected_num_images = len(images)
                np_images = [image.to(torch.float32).numpy() for image in images]

            else:
                expected_num_images = 1 if images.ndim < 4 else images.shape[0]
                np_images = images.to(torch.float32).numpy()
            test_neptune_logger.log_images(images=images, channels_last=channels_last)
            test_neptune_logger.log_images(images=np_images, channels_last=channels_last)

            expected_num_images *= 2  # One set of torch tensors, one set of numpy arrays
            expected_num_images_total += expected_num_images

        test_neptune_logger.post_close()
        assert mock_extend.call_count == 2 * len(image_variants)  # One set of torch tensors, one set of numpy arrays


def test_neptune_logger_doesnt_upload_symlinks(test_neptune_logger, dummy_state):
    with _manage_symlink_creation('test.txt') as symlink_name:
        test_neptune_logger.upload_file(
            state=dummy_state,
            remote_file_name='test_symlink',
            file_path=Path(symlink_name),
        )

    assert not test_neptune_logger.neptune_run.exists(f'{test_neptune_logger._base_namespace}/test_symlink')


@contextlib.contextmanager
def _manage_symlink_creation(file_name: str) -> Generator[str, None, None]:
    with open(file_name, 'w') as f:
        f.write('This is a test file.')

    symlink_name = 'test_symlink.txt'

    os.symlink(file_name, symlink_name)

    assert Path(symlink_name).is_symlink()

    yield symlink_name

    os.remove(symlink_name)
    os.remove(file_name)


def test_neptune_log_image_warns_about_improper_value_range(test_neptune_logger):
    image = np.ones((4, 4)) * 300
    with pytest.warns() as record:
        test_neptune_logger.log_images(images=image)

    assert 'Image value range is not in the expected range of [0.0, 1.0] or [0, 255].' in str(record[0].message)


@patch('composer.loggers.neptune_logger._scale_image_to_0_255', return_value=np.ones((4, 4)))
def test_neptune_log_image_scales_improper_image(mock_scale_img, test_neptune_logger):
    image_variants = [
        np.ones((4, 4)) * 300,
        np.ones((4, 4)) * -1,
        np.identity(4) * 300 - 1,
    ]

    for image in image_variants:
        test_neptune_logger.log_images(images=image)
        mock_scale_img.assert_called_once()
        mock_scale_img.reset_mock()
