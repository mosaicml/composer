# Copyright 2021 MosaicML. All Rights Reserved.

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from composer.models.loss import MIoU, ensure_targets_one_hot, soft_cross_entropy


@pytest.fixture
def block_2D_targets():
    base = torch.arange(4)
    targets = []
    for i in range(4):
        targets.append(torch.roll(base, i).repeat_interleave(2).view(2, 4).repeat_interleave(2, dim=0))
    targets = torch.stack(targets)
    return targets


# (input shape)
@pytest.fixture(params=[(3, 5), (7, 11, 13, 17)])
def fake_input_target_pairs(request):
    input_shape = request.param
    num_classes = input_shape[1]
    reduced_input_shape = list(input_shape)
    reduced_input_shape.pop(1)

    input = torch.randn(input_shape)
    targets_idx = torch.randint(low=-1, high=num_classes, size=reduced_input_shape)
    targets_one_hot = torch.zeros_like(input)
    for i, value in np.ndenumerate(targets_idx):
        i_expanded = list(i)
        if value >= 0:
            i_expanded.insert(1, value)
            targets_one_hot[tuple(i_expanded)] = 1.0
    return input, targets_idx, targets_one_hot


def test_miou(block_2D_targets):
    miou = MIoU(num_classes=4)

    # Test if predictions identical to target equal 1.0
    # TODO: convert to prediction to one-hot
    accurate_prediction = F.one_hot(block_2D_targets, num_classes=4).permute(0, 3, 1, 2)
    miou.update(accurate_prediction, block_2D_targets)
    assert miou.compute() == 100.
    miou.reset()

    # Test if completely incorrect predictions equal 0.0
    inaccurate_prediction = torch.flip(accurate_prediction, dims=(0,))
    miou.update(inaccurate_prediction, block_2D_targets)
    assert miou.compute() == 0.0
    miou.reset()

    # Test if halfway correct predictions is close to 33.3333
    accurateish_prediction = torch.roll(accurate_prediction, shifts=1, dims=2)
    miou.update(accurateish_prediction, block_2D_targets)
    assert torch.isclose(miou.compute(), torch.tensor(33.3333, dtype=torch.double))
    miou.reset()

    # Test if all zeros prediction is equal to 6.25
    all_zeros = torch.zeros(4, 1, 4, 4)
    miou.update(all_zeros, block_2D_targets)
    assert miou.compute() == 6.25
    miou.reset()

    # Test if only one correct sample is equal to 100 * (1/7)
    one_accurate_prediction = inaccurate_prediction.clone()
    one_accurate_prediction[0] = accurate_prediction[0]
    miou.update(one_accurate_prediction, block_2D_targets)
    assert torch.isclose(miou.compute(), torch.tensor(100 / 7, dtype=torch.double))


def test_ensure_targets_one_hot(fake_input_target_pairs):
    input, targets_idx, targets_one_hot = fake_input_target_pairs
    targets_one_hot_test = ensure_targets_one_hot(input, targets_idx)
    torch.testing.assert_allclose(targets_one_hot, targets_one_hot_test)


def test_soft_cross_entropy(fake_input_target_pairs):
    input, targets_idx, targets_one_hot = fake_input_target_pairs
    loss_idx = F.cross_entropy(input, targets_idx, ignore_index=-1)
    loss_soft = soft_cross_entropy(input, targets_one_hot)
    torch.testing.assert_allclose(loss_idx, loss_soft)