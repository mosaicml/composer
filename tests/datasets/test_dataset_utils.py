# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from composer.datasets.utils import pil_image_collate


@pytest.fixture
def num_samples():
    return 4


@pytest.fixture
def image_size():
    return (16, 16)


@pytest.fixture
def pil_image_list(num_samples: int, image_size: Tuple[int, int]):
    return [Image.new(mode='RGB', size=image_size, color=(i, i, i)) for i in range(num_samples)]


@pytest.fixture
def pil_target_list(num_samples: int, image_size: Tuple[int, int]):
    return [Image.new(mode='L', size=image_size, color=i) for i in range(num_samples)]


@pytest.fixture
def correct_image_tensor(num_samples: int, image_size: Tuple[int, int]):
    return torch.arange(num_samples).expand(3, *image_size, -1).permute(3, 0, 1, 2)


@pytest.fixture
def scalar_target_list(num_samples: int):
    return np.arange(num_samples)


def test_scalar_target_collate(pil_image_list: List[Image.Image], scalar_target_list: np.ndarray,
                               correct_image_tensor: torch.Tensor):
    batch = [(img, target) for img, target in zip(pil_image_list, scalar_target_list)]
    image_tensor, target_tensor = pil_image_collate(batch=batch)

    correct_target_tensor = torch.arange(correct_image_tensor.shape[0])

    assert torch.all(image_tensor == correct_image_tensor) and torch.all(target_tensor == correct_target_tensor)


def test_image_target_collate(pil_image_list: List[Image.Image], pil_target_list: List[Image.Image],
                              correct_image_tensor):
    batch = [(img, target) for img, target in zip(pil_image_list, pil_target_list)]
    image_tensor, target_tensor = pil_image_collate(
        batch=batch)  # type: ignore "Image" is incompatible with "ndarray[Unknown, Unknown]"

    assert torch.all(image_tensor == correct_image_tensor) and torch.all(target_tensor == correct_image_tensor[:, 0])
