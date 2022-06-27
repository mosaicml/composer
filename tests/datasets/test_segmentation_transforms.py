# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from PIL import Image

from composer.datasets.ade20k import (PadToSize, PhotometricDistoration, RandomCropPair, RandomHFlipPair,
                                      RandomResizePair)


@pytest.fixture
def size():
    return 16, 16


@pytest.fixture
def sample_pair(size):
    img = Image.new(mode='RGB', size=size)
    target = Image.new(mode='L', size=size)
    return img, target


def test_random_resize(sample_pair, size):
    random_resize_transform = RandomResizePair(min_scale=0.5, max_scale=2.0, base_size=size)

    # Test that the resized image remains within bounds for 10 iterations
    for _ in range(10):
        resized_img, resized_target = random_resize_transform(sample_pair)
        assert resized_img.size == resized_target.size
        assert resized_img.size[0] >= size[0] // 2 and resized_img.size[0] <= size[0] * 2
        assert resized_img.size[1] >= size[1] // 2 and resized_img.size[1] <= size[1] * 2


@pytest.mark.parametrize('crop_size', [(8, 8), (32, 32)])
def test_random_crop(sample_pair, crop_size):
    random_crop_transform = RandomCropPair(crop_size)
    image, target = random_crop_transform(sample_pair)
    assert image.size == target.size
    final_size = min(crop_size[0], sample_pair[0].height), min(crop_size[1], sample_pair[0].width)
    assert final_size == image.size


def test_random_hflip(sample_pair):
    old_image, old_target = np.array(sample_pair[0]), np.array(sample_pair[1])

    # Always flip
    always_hflip_transform = RandomHFlipPair(probability=1.0)
    new_image, new_target = always_hflip_transform(sample_pair)
    new_image, new_target = np.array(new_image), np.array(new_target)
    assert np.allclose(new_image, old_image[:, ::-1]) and np.allclose(new_target, old_target[:, ::-1])

    # Never flip
    always_hflip_transform = RandomHFlipPair(probability=0.0)
    new_image, new_target = always_hflip_transform(sample_pair)
    new_image, new_target = np.array(new_image), np.array(new_target)
    assert np.allclose(new_image, old_image) and np.allclose(new_target, old_target)


@pytest.mark.parametrize('pad_size', [(32, 32), (8, 8)])
def test_pad_transform(sample_pair, pad_size):
    image = sample_pair[0]
    pad_transform = PadToSize(size=pad_size, fill=255)
    padded_image = pad_transform(image)
    final_size = max(pad_size[1], image.width), max(pad_size[0], image.height)
    # Check for correct size and number of padding elements
    assert padded_image.size == final_size

    # Check appropriate amount of padding is used
    padded_image = np.array(padded_image)
    initial_area = image.width * image.height
    final_area = final_size[0] * final_size[1]
    n_channels = padded_image.shape[2]
    pad_volume = n_channels * (final_area - initial_area)
    assert pad_volume == (padded_image == 255).sum()


def test_photometric_distortion(sample_pair):
    old_image = sample_pair[0]
    # Test no transform case
    photometric_transform = PhotometricDistoration(brightness=1.0, contrast=1.0, saturation=1.0, hue=0)
    new_image = photometric_transform(old_image)
    old_image, new_image = np.array(old_image), np.array(new_image)
    assert np.allclose(old_image, new_image)
