# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from composer.metrics import MIoU


@pytest.fixture
def block_2D_targets():
    base = torch.arange(4)
    targets = []
    for i in range(4):
        targets.append(torch.roll(base, i).repeat_interleave(2).view(2, 4).repeat_interleave(2, dim=0))
    targets = torch.stack(targets)
    return targets


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
