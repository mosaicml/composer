# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import imghdr
import json
import os
import pathlib
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Sequence, Type

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from torch.utils.data import DataLoader

from composer.core import Callback, Engine, Event, State
from composer.loggers import InMemoryLogger, Logger, WandBLogger
from composer.trainer import Trainer
from composer.utils import dist
from tests.callbacks.callback_settings import get_cb_kwargs, get_cbs_and_marks
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.datasets import RandomImageDataset
from tests.common.models import SimpleConvModel


class MockArtifact:

    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path

    def download(self, root: pathlib.Path):
        os.makedirs(root)
        shutil.copy2(self.file_path, root)


@pytest.fixture
def test_wandb_logger(tmp_path, dummy_state):
    pytest.importorskip('wandb', reason='wandb is optional')
    os.environ['WANDB_DIR'] = str(tmp_path)
    os.environ['WANDB_MODE'] = 'offline'
    dummy_state.run_name = 'wandb-test-log-image'
    logger = Logger(dummy_state, [])
    wandb_logger = WandBLogger()
    wandb_logger.init(dummy_state, logger)
    return wandb_logger


def test_wandb_log_image(test_wandb_logger):
    pytest.importorskip('wandb', reason='wandb is optional')

    # We group all the image size variants into one test because calling wandb.init() is slow
    image_variants = [
        (torch.rand(4, 4), False),  # 2D image
        (torch.rand(2, 3, 4, 4), False),  # multiple images, not channels last
        (torch.rand(3, 4, 4), False),  # with channels, not channels last
        ([torch.rand(4, 4, 3)], True),  # with channels, channels last
        (torch.rand(2, 4, 4, 3), True),  # multiple images, channels last
        ([torch.rand(4, 4, 3), torch.rand(4, 4, 3)], True)  # multiple images in list
    ]

    expected_num_images_total = 0
    for (images, channels_last) in image_variants:
        if isinstance(images, Sequence):
            expected_num_images = len(images)
            np_images = [image.numpy() for image in images]

        else:
            expected_num_images = 1 if images.ndim < 4 else images.shape[0]
            np_images = images.numpy()
        test_wandb_logger.log_images(images=images, channels_last=channels_last)
        test_wandb_logger.log_images(images=np_images, channels_last=channels_last)
        expected_num_images *= 2  # One set of torch tensors, one set of numpy arrays
        expected_num_images_total += expected_num_images

    test_wandb_logger.post_close()
    img_dir = str(Path(test_wandb_logger.run_dir) / Path('media/images'))
    imgs = [filename for filename in os.listdir(img_dir) if imghdr.what(img_dir + '/' + filename) == 'png']
    actual_num_images = len(imgs)
    assert actual_num_images == expected_num_images_total


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


def test_wandb_log_image_with_masks(test_wandb_logger):
    pytest.importorskip('wandb', reason='wandb is optional')

    # We group all the image size variants into one test because calling comet_experiment.end() is slow
    image_variants = [
        # single image, single mask, channels last
        (torch.randint(0, 256, (4, 4, 3)), {
            'pred': torch.randint(0, 10, (4, 4))
        }, True),
        # multiple images, single mask, channels last
        (torch.rand(2, 4, 4, 3), {
            'pred': torch.randint(0, 10, (2, 4, 4))
        }, True),
        # multiple images, multiple masks, channels last
        (torch.rand(2, 4, 4, 3), {
            'pred': torch.randint(0, 10, (2, 4, 4)),
            'pred2': torch.randint(0, 10, (2, 4, 4))
        }, True),
        # single image, single mask, not channels last
        (torch.randint(0, 256, (3, 4, 4)), {
            'pred': torch.randint(0, 10, (4, 4))
        }, False),
        # multiple images, single mask, not channels last
        (torch.rand(2, 3, 4, 4), {
            'pred': torch.randint(0, 10, (2, 4, 4))
        }, False),
        # multiple images, multiple masks, not channels last
        (torch.rand(2, 3, 4, 4), {
            'pred': torch.randint(0, 10, (2, 4, 4)),
            'pred2': torch.randint(0, 10, (2, 4, 4))
        }, False)
    ]

    expected_num_masks_total = 0
    expected_num_images_total = 0
    for (images, masks, channels_last) in image_variants:
        num_masks = len(masks.keys())
        expected_num_images = 1 if images.ndim < 4 else images.shape[0]
        expected_num_masks = num_masks * expected_num_images

        test_wandb_logger.log_images(images=images, masks=masks, channels_last=channels_last)
        expected_num_images_total += expected_num_images
        expected_num_masks_total += expected_num_masks

    test_wandb_logger.post_close()
    img_dir = str(Path(test_wandb_logger.run_dir) / Path('media/images'))
    imgs = [
        filename for filename in os.listdir(img_dir)
        if not os.path.isdir(img_dir + '/' + filename) and imghdr.what(img_dir + '/' + filename) == 'png'
    ]
    actual_num_images = len(imgs)
    assert actual_num_images == expected_num_images_total

    mask_dir = str(Path(test_wandb_logger.run_dir) / Path('media/images/mask'))
    masks = [filename for filename in os.listdir(mask_dir) if imghdr.what(mask_dir + '/' + filename) == 'png']
    actual_num_masks = len(masks)
    assert actual_num_masks == expected_num_masks_total


@pytest.mark.parametrize('images,masks', [(torch.randint(0, 256, (32, 32, 3)), {
    'pred': torch.randint(0, 10, (32, 32))
})])
def test_wandb_log_image_with_masks_and_table(images, masks, test_wandb_logger):
    wandb = pytest.importorskip('wandb', reason='wandb is optional')

    expected_num_images = 1 if images.ndim < 4 else images.shape[0]

    assert wandb.run is not None
    wandb_run_dir = Path(wandb.run.dir)
    test_wandb_logger.log_images(images=images, masks=masks, channels_last=True, use_table=True)
    test_wandb_logger.post_close()

    wandb_media_dir = wandb_run_dir.parent / Path('files') / Path('media') / Path('table')
    image_table_files = wandb_media_dir.glob('./*.json')

    image_count = 0
    for image_table_file in image_table_files:
        table_columns = json.load(open(image_table_file.absolute()))['data']
        num_images = sum([1 for column in table_columns if column[0] == 'Image'])
        image_count += num_images

    assert image_count == expected_num_images


def test_wandb_log_metrics(test_wandb_logger):
    wandb = pytest.importorskip('wandb', reason='wandb is optional')

    dataset_size = 40
    batch_size = 4

    trainer = Trainer(model=SimpleConvModel(),
                      loggers=test_wandb_logger,
                      train_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      eval_dataloader=DataLoader(RandomImageDataset(size=dataset_size), batch_size),
                      max_duration='1ep')

    trainer.fit()

    wandb_run_dir = Path(wandb.run.dir)
    run_file = wandb_run_dir.parent / Path(f'run-{wandb.run.id}.wandb')

    # delete trainer to force WandBLogger to clean up in post_close
    del trainer

    # Note, it is not clear how to correctly load this file, so we are just loading it as text
    # and searching the text for expected strings
    with open(run_file, encoding='latin-1') as _wandb_file:
        all_run_text = _wandb_file.read()

    train_metrics_accuracy_count = all_run_text.count('metrics/train/MulticlassAccuracy')
    eval_metrics_accuracy_count = all_run_text.count('metrics/eval/MulticlassAccuracy')
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
        for _, data in log_calls:
            # Manually filter out custom W&B data types and tensors, which are allowed, but cannot be json serialized
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
def test_wandb_artifacts(monkeypatch: MonkeyPatch, rank_zero_only: bool, tmp_path: pathlib.Path, dummy_state: State):
    """Test that wandb artifacts logged on rank zero are accessible by all ranks."""
    import wandb
    original_wandb_mode = os.environ.get('WANDB_MODE', None)
    os.environ['WANDB_MODE'] = 'offline'
    # Create the logger
    ctx = pytest.warns(
        UserWarning, match='`rank_zero_only` should be set to False.') if rank_zero_only else contextlib.nullcontext()
    with ctx:
        wandb_logger = WandBLogger(rank_zero_only=rank_zero_only,
                                   log_artifacts=True,
                                   entity='entity',
                                   project='project')
    dummy_state.callbacks.append(wandb_logger)
    logger = Logger(dummy_state, [wandb_logger])
    engine = Engine(dummy_state, logger)
    engine.run_event(Event.INIT)
    wandb_logger.init(dummy_state, logger)

    # Distribute the artifact name from rank 0 to all ranks
    wandb_artifact_name = 'test-wandb-artifact-' + str(uuid.uuid4())
    wandb_artifact_name_list = [wandb_artifact_name]
    dist.broadcast_object_list(wandb_artifact_name_list)
    wandb_artifact_name = wandb_artifact_name_list[0]

    # Have all ranks use the rank 0 save folder
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = pathlib.Path(tmp_paths[0])

    # Create a dummy artifact
    dummy_wandb_artifact_path = save_folder / 'wandb_artifact.txt'
    if dist.get_global_rank() == 0:
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
    downloaded_wandb_artifact_path = save_folder / 'downloaded_wandb_artifact'

    def mock_artifact(*arg, **kwargs):
        return MockArtifact(dummy_wandb_artifact_path)

    monkeypatch.setattr(wandb.Api, 'artifact', mock_artifact)

    def _validate_wandb_artifact():
        wandb_logger.download_file(wandb_artifact_name, str(downloaded_wandb_artifact_path))
        with open(downloaded_wandb_artifact_path, 'r') as f:
            assert f.read() == 'hello!'

    _validate_wandb_artifact()

    # reset wandb mode
    if original_wandb_mode is None:
        del os.environ['WANDB_MODE']
    else:
        os.environ['WANDB_MODE'] = original_wandb_mode
