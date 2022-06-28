# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Tuple

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from composer.algorithms import ColOut
from composer.algorithms.colout.colout import ColOutTransform, colout_batch
from composer.core import Event, State
from composer.loggers import Logger
from tests.common import RandomImageDataset


def verify_shape_image(orig: Image.Image, new: Image.Image, p_row: float, p_col: float) -> None:
    """Verify the shape of a transformed PIL Image."""
    H_o, W_o = orig.height, orig.width
    H_n, W_n = new.height, new.width

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert (H_n, W_n) == (H_t, W_t), f'Image shape mismatch: {(H_n, W_n)} != {(H_t, W_t)}'


def verify_shape_image_pair(orig_sample: Tuple[Image.Image, Image.Image], new_sample: Tuple[Image.Image, Image.Image],
                            p_row: float, p_col: float):
    """Verify the shape of a pair of transformed PIL images."""
    H_o, W_o = orig_sample[0].height, orig_sample[0].width

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert new_sample[0].size == (H_t, W_t), f'Input Image shape mismatch: {new_sample[0].size} != {(H_t, W_t)}'
    assert new_sample[1].size == (H_t, W_t), f'Target image shape mismatch: {new_sample[1].size} != {(H_t, W_t)}'


def verify_shape_tensor(orig: torch.Tensor, new: torch.Tensor, p_row: float, p_col: float) -> None:
    """Verify the shape of a transformed image tensor."""
    C, H_o, W_o = orig.shape

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert new.shape == (C, H_t, W_t), f'Image tensor shape mismatch: {new.shape} != {(C, H_t, W_t)}'


def verify_shape_tensor_pair(orig_sample: Tuple[torch.Tensor, torch.Tensor],
                             new_sample: Tuple[torch.Tensor, torch.Tensor], p_row: float, p_col: float) -> None:
    """Verify the shape of a transformed image tensor."""
    C, H_o, W_o = orig_sample[0].shape

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert new_sample[0].shape == (C, H_t, W_t), f'Input shape mismatch: {new_sample[0].shape} != {(C, H_t, W_t)}'
    assert new_sample[1].shape == (C, H_t, W_t), f'Target shape mismatch: {new_sample[0].shape} != {(C, H_t, W_t)}'


def verify_shape_batch(orig: torch.Tensor, new: torch.Tensor, p_row: float, p_col: float) -> None:
    """Verify the shape of a transformed batch of images."""
    N, C, H_o, W_o = orig.shape

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert new.shape == (N, C, H_t, W_t), f'Image batch shape mismatch: {new.shape} != {(N, C, H_t, W_t)}'


def verify_shape_batch_pair(orig_sample: Tuple[torch.Tensor, torch.Tensor],
                            new_sample: Tuple[torch.Tensor, torch.Tensor], p_row: float, p_col: float) -> None:
    """Verify the shape of a transformed batch of images."""

    N, C, H_o, W_o = orig_sample[0].shape

    H_t = int((1 - p_row) * H_o)
    W_t = int((1 - p_col) * W_o)

    assert new_sample[0].shape == (N, C, H_t, W_t), f'Input shape mismatch: {new_sample[0].shape} != {(N, C, H_t, W_t)}'
    assert new_sample[1].shape == (N, C, H_t,
                                   W_t), f'Target shape mismatch: {new_sample[1].shape} != {(N, C, H_t, W_t)}'


@pytest.fixture(params=[False, True])
def batch(request) -> bool:
    """Algorithm batch parameter."""
    return request.param


@pytest.fixture(params=[0, 0.15])
def p_row(request) -> float:
    """Algorithm p_row parameter."""
    return request.param


@pytest.fixture
def p_col(p_row) -> float:
    """Algorithm p_col parameter."""
    return p_row


@pytest.fixture(params=[1, 3])
def C(request) -> int:
    """Number of image channels.

    Testing BW and RGB.
    """
    return request.param


@pytest.fixture
def H(request) -> int:
    """Default image height."""
    return 32


@pytest.fixture
def W(H) -> int:
    """Default image width (equal to height)"""
    return H


@pytest.fixture
def fake_image(H: int, W: int, C: int) -> Image.Image:
    """Fake PIL Image."""
    return Image.fromarray((255 * np.random.uniform(size=(H, W, C)).squeeze()).astype(np.uint8))


@pytest.fixture
def fake_image_tensor(H: int, W: int, C: int) -> torch.Tensor:
    """Fake image tensor."""
    return torch.rand(C, H, W)


@pytest.fixture
def fake_image_batch(H: int, W: int, C: int) -> torch.Tensor:
    """Fake batch of images."""
    return torch.rand(16, C, H, W)


@pytest.fixture
def colout_algorithm(p_row: float, p_col: float, batch: bool) -> ColOut:
    """Reusable algorithm instance."""
    return ColOut(p_row, p_col, batch)


class TestColOutTransform:

    def test_single_image_drop_size(self, fake_image: Image.Image, p_row: float, p_col: float):
        """Test application to single PIL image."""
        transform = ColOutTransform(p_row, p_col)
        new_image = transform(fake_image)
        assert isinstance(new_image, Image.Image)
        verify_shape_image(fake_image, new_image, p_row, p_col)

    def test_image_pair_drop_size(self, fake_image: Image.Image, p_row: float, p_col: float):
        """Test application to a 2-tuple of PIL images."""
        transform = ColOutTransform(p_row, p_col)
        orig_sample = (fake_image, fake_image)
        new_sample = transform(orig_sample)
        assert isinstance(new_sample, Tuple)
        verify_shape_image_pair(orig_sample, new_sample, p_row, p_col)

    @pytest.mark.parametrize('W', [48])
    def test_rectangular_image(self, fake_image: Image.Image, p_row: float, p_col: float):
        """Test application to a rectangular PIL image."""
        transform = ColOutTransform(p_row, p_col)
        new_image = transform(fake_image)
        verify_shape_image(fake_image, new_image, p_row, p_col)  # type: ignore

    def test_single_image_tensor_drop_size(self, fake_image_tensor: torch.Tensor, p_row: float, p_col: float):
        """Test application to a single torch image tensor."""
        transform = ColOutTransform(p_row, p_col)
        new_image = transform(fake_image_tensor)
        verify_shape_tensor(fake_image_tensor, new_image, p_row, p_col)  # type: ignore

    def test_image_tensor_pair_drop_size(self, fake_image_tensor: torch.Tensor, p_row: float, p_col: float):
        """Test application to a single torch image tensor."""
        transform = ColOutTransform(p_row, p_col)
        orig_sample = (fake_image_tensor, fake_image_tensor)
        new_sample = transform(orig_sample)
        verify_shape_tensor_pair(orig_sample, new_sample, p_row, p_col)  # type: ignore

    def test_reproducibility_image(self, fake_image_tensor: torch.Tensor, p_row: float, p_col: float):
        """Test that transform is reproducible given the same seed."""
        transform_1 = ColOutTransform(p_row, p_col)
        transform_2 = ColOutTransform(p_row, p_col)

        torch.manual_seed(42)
        new_image_1 = transform_1(fake_image_tensor)
        assert isinstance(new_image_1, torch.Tensor)
        torch.manual_seed(42)
        new_image_2 = transform_2(fake_image_tensor)
        assert isinstance(new_image_2, torch.Tensor)

        assert torch.allclose(new_image_1, new_image_2)


class TestColOutFunctional:

    def test_reproducibility_batch(self, fake_image_batch: torch.Tensor, p_row: float, p_col: float):
        """Test that batch augmentation is reproducible given the same seed."""
        transform_1 = functools.partial(colout_batch, p_row=p_row, p_col=p_col)
        transform_2 = functools.partial(colout_batch, p_row=p_row, p_col=p_col)

        torch.manual_seed(42)
        new_batch_1 = transform_1(fake_image_batch)
        torch.manual_seed(42)
        new_batch_2 = transform_2(fake_image_batch)

        assert isinstance(new_batch_1, torch.Tensor)
        assert isinstance(new_batch_2, torch.Tensor)
        assert torch.allclose(new_batch_1, new_batch_2)

    def test_batch_drop_size(self, fake_image_batch: torch.Tensor, p_row: float, p_col: float):
        """Test application to a batch of images."""
        colout = functools.partial(colout_batch, p_row=p_row, p_col=p_col)
        new_batch = colout(fake_image_batch)
        assert isinstance(new_batch, torch.Tensor)
        verify_shape_batch(fake_image_batch, new_batch, p_row, p_col)

    def test_batch_pair_drop_size(self, fake_image_batch: torch.Tensor, p_row: float, p_col: float):
        colout = functools.partial(colout_batch, p_row=p_row, p_col=p_col)
        sample = (fake_image_batch, fake_image_batch)
        new_batch = colout(sample)
        assert isinstance(new_batch, Tuple) and isinstance(new_batch[0], torch.Tensor) and isinstance(
            new_batch[1], torch.Tensor)
        verify_shape_batch_pair(sample, new_batch, p_row, p_col)

    @pytest.mark.parametrize('p_col', [0.05, 0.25])
    def test_rectangle_batch_drop_size(self, fake_image_batch: torch.Tensor, p_row: float, p_col: float):
        """Test that unequal values of p_row and p_col work properly."""
        colout = functools.partial(colout_batch, p_row=p_row, p_col=p_col)
        new_batch = colout(fake_image_batch)
        assert isinstance(new_batch, torch.Tensor)
        verify_shape_batch(fake_image_batch, new_batch, p_row, p_col)


class TestColOutAlgorithm:

    @pytest.mark.parametrize('event,batch', [(Event.AFTER_DATALOADER, True), (Event.FIT_START, False)])
    def test_match_correct(self, event: Event, colout_algorithm: ColOut, minimal_state: State):
        """Algo should match AFTER_DATALOADER if batch else FIT_START."""
        assert colout_algorithm.match(event, minimal_state)

    @pytest.mark.parametrize('event,batch', [(Event.FIT_START, True), (Event.AFTER_DATALOADER, False),
                                             (Event.EPOCH_END, True)])
    def test_match_incorrect(self, event: Event, colout_algorithm: ColOut, minimal_state: State):
        """Algo should NOT match FIT_START if batch else AFTER_DATALOADER."""
        assert not colout_algorithm.match(event, minimal_state)

    @pytest.mark.parametrize('batch', [True])
    def test_apply_batch(self, fake_image_batch: torch.Tensor, colout_algorithm: ColOut, minimal_state: State,
                         empty_logger: Logger):
        """Applies the algorithm to a fake batch."""
        p_row = colout_algorithm.p_row
        p_col = colout_algorithm.p_col

        minimal_state.batch = (fake_image_batch, torch.Tensor())
        colout_algorithm.apply(Event.AFTER_DATALOADER, minimal_state, empty_logger)
        last_input, last_target = minimal_state.batch
        verify_shape_batch(fake_image_batch, last_input, p_row, p_col)
        assert id(minimal_state.batch[1]) == id(last_target)  # Check that the target before and after are the same

    @pytest.mark.parametrize('batch', [True])
    def test_apply_batch_pair(self, fake_image_batch: torch.Tensor, colout_algorithm: ColOut, minimal_state: State,
                              empty_logger: Logger):
        """Applies batch ColOut to 2-tuple of images."""
        p_row = colout_algorithm.p_row
        p_col = colout_algorithm.p_col

        orig_sample = (fake_image_batch, fake_image_batch)
        minimal_state.batch = orig_sample
        colout_algorithm.apply(Event.AFTER_DATALOADER, minimal_state, empty_logger)
        new_sample = minimal_state.batch
        verify_shape_batch_pair(orig_sample, new_sample, p_row, p_col)

    @pytest.mark.parametrize('batch', [False])
    def test_apply_sample(self, colout_algorithm: ColOut, minimal_state: State, empty_logger: Logger):
        """Test that augmentation is added to dataset and functioning properly."""
        p_row = colout_algorithm.p_row
        p_col = colout_algorithm.p_col

        dataset = RandomImageDataset(is_PIL=True)
        dataloader = DataLoader(dataset)

        original_image, _ = dataset[0]
        assert isinstance(original_image, Image.Image)

        minimal_state.set_dataloader(dataloader, 'train')
        colout_algorithm.apply(Event.FIT_START, minimal_state, empty_logger)

        new_image, _ = dataset[0]
        assert isinstance(new_image, Image.Image)

        verify_shape_image(original_image, new_image, p_row, p_col)


@pytest.mark.parametrize('p_row,p_col', [(1.5, 0.15), (0.15, 1.5)])
def test_invalid_hparams(p_row: float, p_col: float):
    """Test that invalid hyperparameters error.

    Ideally this could be caught by the Hparams, but that's not yet supported in yahp.
    """
    with pytest.raises(ValueError):
        ColOut(p_row, p_col, False)
