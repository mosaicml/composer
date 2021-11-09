# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch
from torch.nn import functional as F
import numpy as np

from composer.algorithms import CutMixHparams
from composer.algorithms.cutmix.cutmix import cutmix, rand_bbox
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
    return x_fake, y_fake, indices, n_classes


def validate_cutmix(x, y, indices, x_cutmix, y_cutmix, cutmix_lambda, bbox, n_classes):
    # Create shuffled version of x, y for reference checking
    x_perm = x[indices]
    y_perm = y[indices]

    # Explicitly check that the pixels and labels have been mixed correctly.
    for i in range(x.size(0)):  # Grab N
        # Check every pixel of the input data
        for j in range(x.size(2)):
            for k in range(x.size(3)):
                if (j >= bbox[0] and j < bbox[2]) and (k >= bbox[1] and k < bbox[3]):
                    torch.testing.assert_allclose(x_perm[i, :, j, k], x_cutmix[i, :, j, k])
                else:
                    torch.testing.assert_allclose(x[i, :, j, k], x_cutmix[i, :, j, k])
        # Check the label
        y_onehot = F.one_hot(y[i], num_classes=n_classes)
        y_perm_onehot = F.one_hot(y_perm[i], num_classes=n_classes)
        y_interp = cutmix_lambda * y_onehot + (1 - cutmix_lambda) * y_perm_onehot
        torch.testing.assert_allclose(y_interp, y_cutmix[i])


@pytest.mark.parametrize('alpha', [.2, 1])
class TestCutMix:

    def test_cutmix(self, fake_data, alpha):
        # Generate fake data
        x_fake, y_fake, indices, n_classes = fake_data

        # Get lambda based on alpha hparam
        cutmix_lambda = np.random.beta(alpha, alpha)
        # Get a random bounding box based on cutmix_lambda
        cx = np.random.randint(x_fake.shape[2])
        cy = np.random.randint(x_fake.shape[3])
        bbx1, bby1, bbx2, bby2 = rand_bbox(
                                    W=x_fake.shape[2],
                                    H=x_fake.shape[3],
                                    cutmix_lambda=cutmix_lambda,
                                    cx=cx,
                                    cy=cy)
        bbox = (bbx1, bby1, bbx2, bby2)
        # Adjust lambda
        cutmix_lambda = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_fake.size()[-1] * x_fake.size()[-2]))

        # Apply cutmix
        x_cutmix, y_cutmix = cutmix(
                                x=x_fake,
                                y=y_fake,
                                alpha=1.0,
                                n_classes=n_classes,
                                cutmix_lambda=cutmix_lambda,
                                bbox=bbox,
                                indices=indices)

        # Validate results
        validate_cutmix(
                        x_fake,
                        y_fake,
                        indices,
                        x_cutmix,
                        y_cutmix,
                        cutmix_lambda,
                        bbox,
                        n_classes)

    def test_cutmix_algorithm(self, fake_data, alpha, dummy_state, dummy_logger):
        # Generate fake data
        x_fake, y_fake, _, n_classes = fake_data

        algorithm = CutMixHparams(alpha=alpha).initialize_object()
        state = dummy_state
        state.model = MosaicClassifier
        state.model.num_classes = x_fake.size(1)  # Grab C
        state.batch = (x_fake, y_fake)

        algorithm.apply(Event.INIT, state, dummy_logger)
        # Apply algo, use test hooks to specify indices and override internally generated interpolation lambda for testability
        algorithm.apply(Event.AFTER_DATALOADER, state, dummy_logger)

        x, y = state.batch
        # Validate results
        validate_cutmix(
                        x_fake,
                        y_fake,
                        algorithm.indices,
                        x,
                        y,
                        algorithm.cutmix_lambda,
                        algorithm.bbox,
                        algorithm.num_classes)

@pytest.mark.xfail
def test_cutmix_nclasses(dummy_state, dummy_logger):
    algorithm = CutMixHparams(alpha=1.0).initialize_object()
    state = dummy_state
    state.model = MosaicClassifier
    state.model.num_classes = None  # This should flag AttributeError

    algorithm.apply(Event.AFTER_DATALOADER, state, dummy_logger)


@pytest.mark.run_long
@pytest.mark.timeout(90)
def test_cutmix_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [CutMixHparams(alpha=1.0)]
    train_model(mosaic_trainer_hparams)
