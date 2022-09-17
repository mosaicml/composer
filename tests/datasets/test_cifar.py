# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.datasets import build_cifar10_dataloader, build_synthetic_cifar10_dataloader


@pytest.mark.parametrize('is_train', [False, True])
@pytest.mark.parametrize('synthetic', [False, True])
def test_cifar10_shape_length(is_train, synthetic):
    batch_size = 1

    if synthetic:
        loader = build_synthetic_cifar10_dataloader(batch_size=1, is_train=is_train)
    else:
        loader = build_cifar10_dataloader(datadir='/tmp', batch_size=1, is_train=is_train)

    samples = [_ for _ in loader]
    if is_train:
        assert len(samples) == 50000 // batch_size
    else:
        assert len(samples) == 10000 // batch_size

    assert samples[0][0].shape == (1, 3, 32, 32)
