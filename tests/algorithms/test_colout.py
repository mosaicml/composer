# Copyright 2021 MosaicML. All Rights Reserved.

import functools

import numpy as np
import pytest
import torch
from PIL import Image

from composer.algorithms import ColOut, ColOutHparams
from composer.algorithms.colout.colout import ColOutTransform, batch_colout
from composer.core import Event
from composer.trainer import TrainerHparams
from tests.utils.trainer_fit import train_model


def verify_shape_image(orig: Image.Image, new: Image.Image, p_row: float, p_col: float) -> None:
    """ Verify the shape of a transformed PIL Image """
    H_o, W_o = orig.height, orig.width
    H_n, W_n = new.height, new.width

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert (H_n, W_n) == (H_t, W_t), f"Image shape mismatch: {(H_n, W_n)} != {(H_t, W_t)}"


def verify_shape_tensor(orig: torch.Tensor, new: torch.Tensor, p_row: float, p_col: float) -> None:
    """ Verify the shape of a transformed image tensor """
    C, H_o, W_o = orig.shape

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert new.shape == (C, H_t, W_t), f"Image tensor shape mismatch: {new.shape} != {(C, H_t, W_t)}"


def verify_shape_batch(orig: torch.Tensor, new: torch.Tensor, p_row: float, p_col: float) -> None:
    """ Verify the shape of a transformed batch of images """
    N, C, H_o, W_o = orig.shape

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert new.shape == (N, C, H_t, W_t), f"Image batch shape mismatch: {new.shape} != {(N, C, H_t, W_t)}"


@pytest.fixture(params=[False, True])
def batch(request) -> bool:
    """ Algorithm batch parameter """
    return request.param


@pytest.fixture(params=[0, 0.15])
def p_row(request) -> float:
    """ Algorithm p_row parameter """
    return request.param


@pytest.fixture
def p_col(p_row) -> float:
    """ Algorithm p_col parameter """
    return p_row


@pytest.fixture(params=[1, 3])
def C(request) -> int:
    """ Number of image channels. Testing BW and RGB. """
    return request.param


@pytest.fixture
def H(request) -> int:
    """ Default image height """
    return 32


@pytest.fixture
def W(H) -> int:
    """ Default image width (equal to height) """
    return H


@pytest.fixture
def fake_image(H, W, C) -> Image.Image:
    """ Fake PIL Image """
    rng = np.random.RandomState(0)
    return Image.fromarray((255 * rng.uniform(size=(H, W, C)).squeeze()).astype(np.uint8))


@pytest.fixture
def fake_image_tensor(H, W, C) -> torch.Tensor:
    """ Fake image tensor """
    torch.manual_seed(0)
    return torch.rand(C, H, W)


@pytest.fixture
def fake_image_batch(H, W, C) -> torch.Tensor:
    """ Fake batch of images """
    torch.manual_seed(0)
    return torch.rand(16, C, H, W)


@pytest.fixture
def dummy_hparams(p_row, p_col, batch) -> ColOutHparams:
    """ Dummy hparams """
    return ColOutHparams(p_row, p_col, batch)


@pytest.fixture
def dummy_algorithm(dummy_hparams) -> ColOut:
    """ Reusable algorithm instance """
    return ColOut(dummy_hparams.p_row, dummy_hparams.p_col, dummy_hparams.batch)


def test_single_image_drop_size(fake_image, p_row, p_col):
    """ Test application to single PIL image """
    transform = ColOutTransform(p_row, p_col)
    new_image = transform(fake_image)
    verify_shape_image(fake_image, new_image, p_row, p_col)  # type: ignore


@pytest.mark.parametrize("W", [48])
def test_rectangular_image(fake_image, p_row, p_col):
    """ Test application to a rectangular PIL image """
    transform = ColOutTransform(p_row, p_col)
    new_image = transform(fake_image)
    verify_shape_image(fake_image, new_image, p_row, p_col)  # type: ignore


def test_single_image_tensor_drop_size(fake_image_tensor, p_row, p_col):
    """ Test application to a single torch image tensor """
    transform = ColOutTransform(p_row, p_col)
    new_image = transform(fake_image_tensor)
    verify_shape_tensor(fake_image_tensor, new_image, p_row, p_col)  # type: ignore


def test_reproducibility_image(fake_image_tensor, p_row, p_col):
    """ Test that transform is reproducible given the same seed """
    transform_1 = ColOutTransform(p_row, p_col)
    transform_2 = ColOutTransform(p_row, p_col)

    torch.manual_seed(42)
    new_image_1 = transform_1(fake_image_tensor)
    torch.manual_seed(42)
    new_image_2 = transform_2(fake_image_tensor)

    assert torch.allclose(new_image_1, new_image_2)  # type: ignore


def test_reproducibility_batch(fake_image_batch, p_row, p_col):
    """ Test that batch augmentation is reproducible given the same seed """
    transform_1 = functools.partial(batch_colout, p_row=p_row, p_col=p_col)
    transform_2 = functools.partial(batch_colout, p_row=p_row, p_col=p_col)

    torch.manual_seed(42)
    new_batch_1 = transform_1(fake_image_batch)
    torch.manual_seed(42)
    new_batch_2 = transform_2(fake_image_batch)

    assert torch.allclose(new_batch_1, new_batch_2)  # type: ignore


def test_batch_drop_size(fake_image_batch, p_row, p_col):
    """ Test application to a batch of images """
    colout = functools.partial(batch_colout, p_row=p_row, p_col=p_col)
    new_batch = colout(fake_image_batch)
    verify_shape_batch(fake_image_batch, new_batch, p_row, p_col)


@pytest.mark.parametrize("p_col", [0.05, 0.25])
def test_rectangle_batch_drop_size(fake_image_batch, p_row, p_col):
    """ Test that unequal values of p_row and p_col work properly """
    colout = functools.partial(batch_colout, p_row=p_row, p_col=p_col)
    new_batch = colout(fake_image_batch)
    verify_shape_batch(fake_image_batch, new_batch, p_row, p_col)


@pytest.mark.parametrize("p_row,p_col", [(1.5, 0.15), (0.15, 1.5)])
def test_invalid_hparams(p_row, p_col):
    """ Test that invalid hyperparameters error.
    Ideally this could be caught by the Hparams, but that's not yet supported in yahp.
    """
    with pytest.raises(ValueError):
        ColOut(p_row, p_col, False)


@pytest.mark.parametrize("event,batch", [(Event.AFTER_DATALOADER, True), (Event.TRAINING_START, False)])
def test_match_correct(event, dummy_algorithm, dummy_state):
    """ Algo should match AFTER_DATALOADER if batch else TRAINING_START """
    assert dummy_algorithm.match(event, dummy_state)


@pytest.mark.parametrize("event,batch", [(Event.TRAINING_START, True), (Event.AFTER_DATALOADER, False),
                                         (Event.EPOCH_END, True)])
def test_match_incorrect(event, dummy_algorithm, dummy_state):
    """ Algo should NOT match TRAINING_START if batch else AFTER_DATALOADER """
    assert not dummy_algorithm.match(event, dummy_state)


@pytest.mark.parametrize("batch", [True])
def test_apply_batch(fake_image_batch, dummy_algorithm, dummy_state, dummy_logger):
    """ Apply the algorithm to a fake batch """
    p_row = dummy_algorithm.hparams.p_row
    p_col = dummy_algorithm.hparams.p_col

    dummy_state.batch = (fake_image_batch, None)
    dummy_algorithm.apply(Event.AFTER_DATALOADER, dummy_state, dummy_logger)
    last_input, _ = dummy_state.batch
    verify_shape_batch(fake_image_batch, last_input, p_row, p_col)


@pytest.mark.parametrize("batch", [False])
def test_apply_sample(dummy_algorithm, dummy_state, dummy_train_dataloader, dummy_logger):
    """ Test that augmentation is added to dataset and functioning properly """
    p_row = dummy_algorithm.hparams.p_row
    p_col = dummy_algorithm.hparams.p_col

    dset = dummy_train_dataloader.dataset
    orig, _ = dset[0]

    dummy_state.train_dataloader = dummy_train_dataloader
    dummy_algorithm.apply(Event.TRAINING_START, dummy_state, dummy_logger)
    new, _ = dset[0]

    verify_shape_tensor(orig, new, p_row, p_col)


@pytest.mark.run_long
@pytest.mark.timeout(90)
def test_colout_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [ColOutHparams(p_row=0.15, p_col=0.15, batch=True)]
    train_model(mosaic_trainer_hparams)
