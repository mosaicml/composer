# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.algorithms.cutout.cutout import CutOut, _generate_mask
from composer.core import Event


# Box validaton checks for a continuous rectangle, cannot handle multiple/coalesced boxes along x, y dimensions
def _box_validate(mask_box: torch.Tensor) -> None:
    # Box is not contiguous if there are any 0's in the tensor
    box_is_contiguous = not (0 in mask_box)
    assert box_is_contiguous


def _find_box(img_2d: torch.Tensor) -> torch.Tensor:
    height, width = img_2d.size()

    # Generate helper tensors
    ones = torch.ones(height, width)
    zeros = torch.zeros(height, width)

    # Find the box
    # First create h x w filter populated with ones where it thinks there's a box, then find coordinates of ones
    filter_box = torch.where(img_2d == 0, ones, zeros)
    box_x, box_y = torch.nonzero(filter_box, as_tuple=True)  # Find all points where filter_box is 1

    # Check for no box found
    if ((box_x.size()[0], box_y.size()[0]) == (0, 0)):
        # Return valid box as this is possible when cutout_length=1
        return torch.ones(1, 1)
    else:
        # Returns box defined by longest diagonal
        return filter_box[box_x[0]:box_x[-1] + 1, box_y[0]:box_y[-1] + 1]


def check_box(batch_size, channels, input):
    for b in range(batch_size):
        for c in range(channels):
            mask_box = _find_box(input[b, c, :, :])
            _box_validate(mask_box)


# Test square, rectangle inputs
@pytest.fixture(params=[(1, 1, 16, 16), (1, 1, 16, 32)])
def tensor_sizes(request):
    return request.param


# cutout_length=1 won't 0 out (no box is valid)
# cutout_length=3 should produce 2x2 box due to floor division except when boundary clipping
# cutout_length=4 should produce 4x4 box due except when boundary clipping
# cutout_length=0.5 should produce a box with half the side length of the input
@pytest.fixture(params=[1, 3, 4])
def cutout_length_pixels(request):
    return request.param


# Check corners, edges and middle
@pytest.fixture(params=[(0, 0), (16, 0), (0, 16), (16, 16), (7, 7)])
def anchors(request):
    return request.param


def test_cutout_mask(tensor_sizes, cutout_length_pixels, anchors):

    batch_size, channels, height, width = tensor_sizes
    x, y = anchors

    test_mask = torch.ones(tensor_sizes)
    test_mask = _generate_mask(mask=test_mask, width=width, height=height, x=x, y=y, cutout_length=cutout_length_pixels)

    check_box(batch_size, channels, test_mask)


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('channels', [1, 4])
@pytest.mark.parametrize('height', [32, 64])
@pytest.mark.parametrize('width', [32, 71])
@pytest.mark.parametrize('cutout_length', [0.25, 0.5])
@pytest.mark.parametrize('uniform_sampling', [True, False])
def test_cutout_algorithm(batch_size, channels, height, width, cutout_length, empty_logger, minimal_state,
                          uniform_sampling):

    # Initialize input tensor
    #   - Add bias to prevent 0. pixels, causes check_box() to fail since based on search for 0's
    #   - Real data can have 0. pixels but this will not affect cutout algorithm since mask is generated independent of input data
    input = torch.rand((batch_size, channels, height, width)) + 1

    # Fix cutout_num_holes=1, mask generation is additive and box validation isn't smart enough to detect multiple/coalesced boxes
    algorithm = CutOut(num_holes=1, length=cutout_length, uniform_sampling=uniform_sampling)
    state = minimal_state
    state.batch = (input, torch.Tensor())

    algorithm.apply(Event.AFTER_DATALOADER, state, empty_logger)

    input, _ = state.batch
    check_box(batch_size, channels, input)
