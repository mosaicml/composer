# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch

from composer.algorithms import MixUpHparams
from composer.algorithms.mixup.mixup import _gen_mixing_coef, mixup_batch
from composer.core import Event
from composer.models.base import ComposerClassifier


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


def validate_mixup_batch(x, y, indices, x_mix, int_lamb):
    # Create shuffled version of y_fake for reference checking
    y_perm = y[indices]

    # Explicitly check that the batches and labels have been mixed correctly.
    for i in range(x.size(0)):  # Grab N
        j = indices[i]
        # Check the input data
        x_mix_test = (1 - int_lamb) * x[i] + int_lamb * x[j]
        torch.testing.assert_allclose(x_mix_test, x_mix[i])
        # Check the label
        perm_label = y[j]
        torch.testing.assert_allclose(perm_label, y_perm[i])


@pytest.mark.parametrize('alpha', [.2, 1])
class TestMixUp:

    def test_mixup_batch(self, fake_data, alpha):
        # Generate fake data
        x_fake, y_fake, indices = fake_data

        # Get interpolation lambda based on alpha hparam
        mixing_coef = _gen_mixing_coef(alpha)

        # Apply mixup
        x_mix, _, _ = mixup_batch(
            x_fake,
            y_fake,
            mixing=mixing_coef,
            num_classes=x_fake.size(1),  # Grab C
            indices=indices)

        # Validate results
        validate_mixup_batch(x_fake, y_fake, indices, x_mix, mixing_coef)

    def test_mixup_algorithm(self, fake_data, alpha, minimal_state, empty_logger):
        # Generate fake data
        x_fake, y_fake, _ = fake_data

        algorithm = MixUpHparams(alpha=alpha, num_classes=x_fake.size(1)).initialize_object()
        state = minimal_state
        state.model = ComposerClassifier(torch.nn.Flatten())
        state.batch = (x_fake, y_fake)

        # Apply algo, use test hooks to specify indices and override internally generated interpolation lambda for testability
        algorithm.apply(Event.AFTER_DATALOADER, state, empty_logger)

        x, _ = state.batch
        # Use algorithm generated indices and mixing_coef for validation
        validate_mixup_batch(x_fake, y_fake, algorithm.indices, x, algorithm.mixing)
