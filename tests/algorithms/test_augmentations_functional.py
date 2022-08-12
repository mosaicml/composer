# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Tuple, Union, cast

import numpy as np
import pytest
import torch
from PIL.Image import Image as PillowImage
from PIL.Image import fromarray

from composer.algorithms.utils.augmentation_common import image_as_type
from composer.functional import augmix_image, colout_batch, cutout_batch, randaugment_image

AnyImage = Union[torch.Tensor, PillowImage]
InputAugFunction = Callable[[AnyImage], AnyImage]


def _input_image(img_type: str, dtype: torch.dtype) -> AnyImage:
    rng = np.random.default_rng(123)
    torch.manual_seed(123)
    N, H, W, C = 4, 6, 5, 3

    if img_type == 'pillow':
        ints = rng.integers(256, size=(H, W, C)).astype(np.uint8)
        return fromarray(ints, mode='RGB')
    elif dtype == torch.uint8:
        if img_type == 'single_tensor':
            return torch.randint(256, size=(C, H, W)).to(dtype=torch.uint8)
        return torch.randint(256, size=(N, C, H, W)).to(dtype=torch.uint8)
    elif dtype in (torch.float16, torch.float, torch.float64):
        if img_type == 'single_tensor':
            return torch.rand(size=(C, H, W)).to(dtype=dtype)
        return torch.rand(size=(N, C, H, W)).to(dtype=dtype)
    else:
        raise ValueError(f'Invalid dtype: {dtype}')


def _input_output_pair(img_type: str, img_dtype: torch.dtype, f_aug: InputAugFunction) -> Tuple[AnyImage, AnyImage]:
    img = _input_image(img_type, dtype=img_dtype)
    return img, f_aug(img)


@pytest.fixture(params=(torch.uint8, torch.float16, torch.float, torch.float64))
def img_dtype(request) -> torch.dtype:
    return request.param


@pytest.mark.parametrize('img_type', ['pillow', 'single_tensor', 'batch_tensor'])
@pytest.mark.parametrize('f_aug', [colout_batch, cutout_batch, augmix_image, randaugment_image])
def test_batch_augmentation_funcs_preserve_type(img_type: str, img_dtype: torch.dtype, f_aug: InputAugFunction):
    img, out = _input_output_pair(img_type, img_dtype, f_aug)
    assert type(out) == type(img)


@pytest.mark.parametrize('img_type', ['pillow', 'single_tensor', 'batch_tensor'])
@pytest.mark.parametrize('f_aug', [cutout_batch, augmix_image, randaugment_image])  # colout changes shape
def test_batch_augmentation_funcs_preserve_shape(img_type: str, img_dtype: torch.dtype, f_aug: InputAugFunction):
    img, out = _input_output_pair(img_type, img_dtype, f_aug)
    if img_type == 'pillow':
        img = cast(PillowImage, img)
        out = cast(PillowImage, out)
        img = image_as_type(img, torch.Tensor)
        out = image_as_type(out, torch.Tensor)
    assert isinstance(img, torch.Tensor)
    assert isinstance(out, torch.Tensor)
    assert out.shape == img.shape
