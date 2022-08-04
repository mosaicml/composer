# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from composer.algorithms import CutMix
from composer.algorithms.cutmix.cutmix import _rand_bbox, cutmix_batch
from composer.core import Event
from composer.models import ComposerClassifier


# (N, C, d1, d2, num_classes)
@pytest.fixture(params=[(7, 11, 3, 5, 10)])
def fake_data(request):
    # Generate some fake data
    N, C, d1, d2, num_classes = request.param
    x_fake = torch.randn(N, C, d1, d2)
    y_fake = torch.randint(num_classes, size=(N,))
    indices = torch.randperm(N)
    return x_fake, y_fake, indices


def validate_cutmix(x, y, indices, x_cutmix, y_perm, bbox):
    # Create shuffled version of x, y for reference checking
    x_perm = x[indices]
    y_perm_ref = y[indices]

    # Explicitly check that the pixels and labels have been mixed correctly.
    for i in range(x.size(0)):  # Grab N
        # Check every pixel of the input data
        for j in range(x.size(2)):
            for k in range(x.size(3)):
                if (j >= bbox[0] and j < bbox[2]) and (k >= bbox[1] and k < bbox[3]):
                    torch.testing.assert_close(x_perm[i, :, j, k], x_cutmix[i, :, j, k])
                else:
                    torch.testing.assert_close(x[i, :, j, k], x_cutmix[i, :, j, k])
    # Check the label
    torch.testing.assert_close(y_perm_ref, y_perm)


@pytest.mark.parametrize('alpha', [0.2, 1])
@pytest.mark.parametrize('uniform_sampling', [True, False])
@pytest.mark.parametrize('interpolate_loss', [True, False])
class TestCutMix:

    def test_cutmix(self, fake_data, alpha, uniform_sampling, interpolate_loss):
        # Generate fake data
        x_fake, y_fake, indices = fake_data

        # Get lambda based on alpha hparam
        cutmix_lambda = np.random.beta(alpha, alpha)
        # Get a random bounding box based on cutmix_lambda
        cx = np.random.randint(x_fake.shape[2])
        cy = np.random.randint(x_fake.shape[3])
        bbx1, bby1, bbx2, bby2 = _rand_bbox(W=x_fake.shape[2],
                                            H=x_fake.shape[3],
                                            cutmix_lambda=cutmix_lambda,
                                            cx=cx,
                                            cy=cy,
                                            uniform_sampling=uniform_sampling)
        bbox = (bbx1, bby1, bbx2, bby2)
        # Adjust lambda
        cutmix_lambda = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_fake.size()[-1] * x_fake.size()[-2]))

        # Apply cutmix
        x_cutmix, y_perm, _, _ = cutmix_batch(x_fake,
                                              y_fake,
                                              alpha=1.0,
                                              bbox=bbox,
                                              indices=indices,
                                              uniform_sampling=uniform_sampling)

        # Validate results
        validate_cutmix(x=x_fake, y=y_fake, indices=indices, x_cutmix=x_cutmix, y_perm=y_perm, bbox=bbox)

    def test_cutmix_algorithm(self, fake_data, alpha, uniform_sampling, minimal_state, empty_logger, interpolate_loss):
        # Generate fake data
        x_fake, y_fake, _ = fake_data

        algorithm = CutMix(alpha=alpha, uniform_sampling=uniform_sampling, interpolate_loss=interpolate_loss)
        state = minimal_state
        state.model = ComposerClassifier(torch.nn.Flatten())
        state.batch = (x_fake, y_fake)

        # Apply algo, use test hooks to specify indices and override internally generated interpolation lambda for testability
        algorithm.apply(Event.BEFORE_FORWARD, state, empty_logger)

        x, _ = state.batch
        y_perm = algorithm._permuted_target
        # Validate results
        validate_cutmix(x=x_fake, y=y_fake, indices=algorithm._indices, x_cutmix=x, y_perm=y_perm, bbox=algorithm._bbox)
