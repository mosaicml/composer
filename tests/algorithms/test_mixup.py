# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch

from composer.algorithms import MixUpHparams
from composer.algorithms.mixup.mixup import gen_interpolation_lambda, mixup_batch
from composer.core.types import Event
from composer.models.base import MosaicClassifier
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


# (N, C, d1, d2, n_classes)
@pytest.fixture(params=[(7, 11, 3, 5, 10)])
def fake_data(request):
    # Generate some fake data
    N, C, d1, d2, n_classes = request.param
    torch.manual_seed(0)
    x_fake = torch.randn(N, C, d1, d2)
    y_fake = torch.randint(n_classes, size=(N,))
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
        interpolation_lambda = gen_interpolation_lambda(alpha)

        # Apply mixup
        x_mix, y_mix, _ = mixup_batch(
            x=x_fake,
            y=y_fake,
            interpolation_lambda=interpolation_lambda,
            n_classes=x_fake.size(1),  # Grab C
            indices=indices)

        # Validate results
        validate_mixup_batch(x_fake, y_fake, indices, x_mix, interpolation_lambda)

    def test_mixup_algorithm(self, fake_data, alpha, dummy_state, dummy_logger):
        # Generate fake data
        x_fake, y_fake, _ = fake_data

        algorithm = MixUpHparams(alpha=alpha).initialize_object()
        state = dummy_state
        state.model = MosaicClassifier
        state.model.num_classes = x_fake.size(1)  # Grab C
        state.batch = (x_fake, y_fake)

        algorithm.apply(Event.INIT, state, dummy_logger)
        # Apply algo, use test hooks to specify indices and override internally generated interpolation lambda for testability
        algorithm.apply(Event.AFTER_DATALOADER, state, dummy_logger)

        x, _ = state.batch
        # Use algorithm generated indices and interpolation_lambda for validation
        validate_mixup_batch(x_fake, y_fake, algorithm.indices, x, algorithm.interpolation_lambda)


@pytest.mark.xfail
def test_mixup_nclasses(dummy_state, dummy_logger):
    algorithm = MixUpHparams(alpha=0.2).initialize_object()
    state = dummy_state
    state.model = MosaicClassifier
    state.model.num_classes = None  # This should flag AttributeError

    algorithm.apply(Event.AFTER_DATALOADER, state, dummy_logger)


@pytest.mark.run_long
@pytest.mark.timeout(90)
def test_mixup_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [MixUpHparams(alpha=0.2)]
    train_model(mosaic_trainer_hparams)
