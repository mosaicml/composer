# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import imghdr
import json
import os
import pathlib
import pickle
import uuid
from pathlib import Path
from typing import Sequence, Type

import pytest
import torch
from torch.utils.data import DataLoader

from composer.core import Engine, Event
from composer.core.callback import Callback
from composer.core.state import State
from composer.loggers import InMemoryLogger
from composer.loggers.logger import Logger
from composer.loggers.wandb_logger import WandBLogger
from composer.trainer import Trainer
from composer.utils import dist, retry
from tests.callbacks.callback_settings import get_cb_kwargs, get_cbs_and_marks
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.fixture
def test_wandb_logger(tmp_path, dummy_state):
    pytest.importorskip('wandb', reason='wandb is optional')
    os.environ['WANDB_DIR'] = str(tmp_path)
    os.environ['WANDB_MODE'] = 'offline'
    dummy_state.run_name = 'wand-test-log-image'
    logger = Logger(dummy_state, [])
    wandb_logger = WandBLogger()
    wandb_logger.init(dummy_state, logger)
    return wandb_logger


@pytest.mark.parametrize('images,channels_last', [(torch.rand(32, 32), False), (torch.rand(5, 3, 32, 32), False),
                                                  (torch.rand(3, 32, 32), False), (torch.rand(8, 32, 32, 3), True),
                                                  ([torch.rand(32, 32, 3)], True),
                                                  ([torch.rand(32, 32, 3), torch.rand(32, 32, 3)], True)])
def test_wandb_log_image(tmp_path: pathlib.Path, images, channels_last, test_wandb_logger):
    pytest.importorskip('wandb', reason='wandb is optional')
    if isinstance(images, Sequence):
        expected_num_images = len(images)
        np_images = [image.numpy() for image in images]

    else:
        expected_num_images = 1 if images.ndim < 4 else images.shape[0]
        np_images = images.numpy()
    test_wandb_logger.log_images(images=images, channels_last=channels_last)
    test_wandb_logger.log_images(images=np_images, channels_last=channels_last)
    test_wandb_logger.post_close()
    img_dir = str(Path(test_wandb_logger.run_dir) / Path('media/images'))
    expected_num_images *= 2  # One set of torch tensors, one set of numpy arrays
    imgs = [filename for filename in os.listdir(img_dir) if imghdr.what(img_dir + '/' + filename) == 'png']
    actual_num_images = len(imgs)
    assert actual_num_images == expected_num_images


@pytest.mark.parametrize(
    'images,channels_last',
    [
        (torch.rand(32), False),
        (torch.rand(32, 0), False),  # Has zero in dimension.
        (torch.rand(4, 4, 8, 32, 32), False),  # > 4 dim.
        ([torch.rand(4, 32, 32, 3)], True),
    ])  # sequence > 3 dim.
def test_wandb_ml_log_image_errors_out(test_wandb_logger, images, channels_last):
    pytest.importorskip('wandb', reason='wandb is optional')
    with pytest.raises(ValueError):
        test_wandb_logger.log_images(images, channels_last=channels_last)


@pytest.mark.parametrize('callback_cls', get_cbs_and_marks(callbacks=True))
def test_logged_data_is_json_serializable(callback_cls: Type[Callback]):
    """Test that all logged data is json serializable, which is a requirement to use wandb."""
    pytest.importorskip('wandb', reason='wandb is optional')
    from wandb.sdk.data_types.base_types.wb_value import WBValue
    callback_kwargs = get_cb_kwargs(callback_cls)
    callback = callback_cls(**callback_kwargs)
    logger = InMemoryLogger()  # using an in memory logger to manually validate json serializability
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=2,
        max_duration='1ep',
        callbacks=callback,
        loggers=logger,
    )
    trainer.fit()

    for log_calls in logger.data.values():
        for timestamp, data in log_calls:
            del timestamp  # unused
            # manually filter out custom W&B data types and tensors, which are allowed, but cannot be json serialized
            if isinstance(data, (WBValue, torch.Tensor)):
                continue
            json.dumps(data)


def test_wandb_is_pickleable_when_disabled(dummy_state: State):
    pytest.importorskip('wandb', reason='wandb is optional')
    original_wandb_mode = os.environ.get('WANDB_MODE', None)
    os.environ['WANDB_MODE'] = 'disabled'
    wandb_logger = WandBLogger()

    # Need to initialize WandbLogger before calling .state_dict()
    dummy_state.callbacks.append(wandb_logger)
    logger = Logger(dummy_state, [wandb_logger])
    engine = Engine(dummy_state, logger)
    engine.run_event(Event.INIT)

    # Just make sure this doesn't crash due to wandb.sdk.lib.disabled.RunDisabled not being pickleable
    pickle.loads(pickle.dumps(wandb_logger.state_dict()))

    # reset wandb mode
    if original_wandb_mode is None:
        del os.environ['WANDB_MODE']
    else:
        os.environ['WANDB_MODE'] = original_wandb_mode


@pytest.mark.world_size(2)
@pytest.mark.parametrize('rank_zero_only', [True, False])
@pytest.mark.skip('This test needs to be refactored to use a Mock API interface.')
def test_wandb_artifacts(rank_zero_only: bool, tmp_path: pathlib.Path, dummy_state: State):
    """Test that wandb artifacts logged on rank zero are accessible by all ranks."""
    pytest.importorskip('wandb', reason='wandb is optional')
    # Create the logger
    ctx = pytest.warns(
        UserWarning, match='`rank_zero_only` should be set to False.') if rank_zero_only else contextlib.nullcontext()
    with ctx:
        wandb_logger = WandBLogger(
            rank_zero_only=rank_zero_only,
            log_artifacts=True,
        )
    dummy_state.callbacks.append(wandb_logger)
    logger = Logger(dummy_state, [wandb_logger])
    engine = Engine(dummy_state, logger)
    engine.run_event(Event.INIT)

    # Distribute the artifact name from rank 0 to all ranks
    wandb_artifact_name = 'test-wandb-artifact-' + str(uuid.uuid4())
    wandb_artifact_name_list = [wandb_artifact_name]
    dist.broadcast_object_list(wandb_artifact_name_list)
    wandb_artifact_name = wandb_artifact_name_list[0]

    if dist.get_global_rank() == 0:
        # Create a dummy artifact
        dummy_wandb_artifact_path = tmp_path / 'wandb_artifact.txt'
        with open(dummy_wandb_artifact_path, 'w+') as f:
            f.write('hello!')

        # Log a wandb artifact if rank zero
        logger.upload_file(
            file_path=dummy_wandb_artifact_path,
            remote_file_name=wandb_artifact_name,
        )

    # Wait for rank 0 queue the file upload
    dist.barrier()

    # Attempt to retrieve the artifact on all ranks
    downloaded_wandb_artifact_path = tmp_path / 'downloaded_wandb_artifact'

    @retry(FileNotFoundError, num_attempts=6)  # 6 attempts is ~2^(6-1) seconds max wait
    def _validate_wandb_artifact():
        wandb_logger.download_file(wandb_artifact_name, str(downloaded_wandb_artifact_path))
        with open(downloaded_wandb_artifact_path, 'r') as f:
            assert f.read() == 'hello!'

    _validate_wandb_artifact()
