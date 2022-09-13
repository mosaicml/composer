# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.datasets.cifar import build_cifar10_dataloader, build_synthetic_cifar10_dataloader


@pytest.mark.skip()
@pytest.mark.parametrize('is_train', [False, True])
@pytest.mark.parametrize('synthetic', [False, True])
def test_cifar10_shape_length(is_train, synthetic):
    f_factory = build_cifar10_dataloader if not synthetic else build_synthetic_cifar10_dataloader
    batch_size = 1
    loader = f_factory(datadir='/tmp', batch_size=batch_size, is_train=is_train, num_workers=0)

    shape = (3, 32, 32)
    samples = [_ for _ in loader]
    if is_train:
        assert len(samples) == 50000 // batch_size
    else:
        assert len(samples) == 10000 // batch_size
    as_tensor = torch.stack(samples)
    assert tuple(as_tensor.shape[1:]) == shape
