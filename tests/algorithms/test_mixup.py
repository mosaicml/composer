# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.algorithms import MixUp
from composer.algorithms.mixup.mixup import _gen_mixing_coef, mixup_batch
from composer.core import Event
from composer.models import ComposerClassifier


# (N, C, d1, d2, num_classes)
@pytest.fixture(params=[(7, 11, 3, 5, 10)])
def fake_data(request):
    # Generate some fake data
    N, C, d1, d2, num_classes = request.param
    torch.manual_seed(0)
    x_fake = torch.randn(N, C, d1, d2)
    y_fake = torch.randint(num_classes, size=(N,))
    indices = torch.randperm(N)
    return x_fake, y_fake, indices


def validate_mixup_batch(x, y, indices, x_mix, y_perm, mixing):
    # Explicitly check that the batches and labels have been mixed correctly.
    for i in range(x.size(0)):  # Grab N
        j = indices[i]
        # Check the input data
        x_mix_test = (1 - mixing) * x[i] + mixing * x[j]
        torch.testing.assert_close(x_mix_test, x_mix[i])
        # Check the label
        perm_label = y[j]
        torch.testing.assert_close(perm_label, y_perm[i])


@pytest.mark.parametrize('alpha', [.2, 1])
@pytest.mark.parametrize('interpolate_loss', [True, False])
class TestMixUp:

    def test_mixup_batch(self, fake_data, alpha, interpolate_loss):
        # Generate fake data
        x_fake, y_fake, indices = fake_data

        # Get interpolation lambda based on alpha hparam
        mixing = _gen_mixing_coef(alpha)

        # Apply mixup
        x_mix, y_perm, _ = mixup_batch(x_fake, y_fake, mixing=mixing, indices=indices)

        # Validate results
        validate_mixup_batch(x_fake, y_fake, indices, x_mix, y_perm, mixing)

    def test_mixup_algorithm(self, fake_data, alpha, interpolate_loss, minimal_state, empty_logger):
        # Generate fake data
        x_fake, y_fake, _ = fake_data

        algorithm = MixUp(alpha=alpha, interpolate_loss=interpolate_loss)
        state = minimal_state
        state.model = ComposerClassifier(torch.nn.Flatten())
        state.batch = (x_fake, y_fake)

        # Apply algo, use test hooks to specify indices and override internally generated interpolation lambda for testability
        algorithm.apply(Event.BEFORE_FORWARD, state, empty_logger)

        x, _ = state.batch
        # Use algorithm generated indices and mixing_coef for validation
        validate_mixup_batch(x_fake, y_fake, algorithm.indices, x, algorithm.permuted_target, algorithm.mixing)
