# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from composer.algorithms import label_smoothing
from composer.algorithms.label_smoothing import LabelSmoothingHparams
from composer.core.types import Event
from composer.models import loss
from composer.trainer.trainer_hparams import TrainerHparams
from tests.utils.trainer_fit import train_model


def _generate_tensors_classification(batch_size: int, num_classes: int):
    """
    Helper functions to generate input, target pairs
    for image classification (1d indicies)
    """
    N = batch_size
    C = num_classes

    target_indicies = torch.randint(0, C, [N])
    target_onehot = F.one_hot(target_indicies, num_classes=C)
    input = F.softmax(torch.randn((N, C)), dim=1)

    return (input, target_indicies, target_onehot)


def _generate_tensors_segmentation(batch_size: int, num_classes: int, H: int, W: int):
    """
    Helper functions to generate input, target pairs
    for image segmentation (2d indicies)
    """
    N = batch_size
    C = num_classes

    target_indicies = torch.randint(0, C, (N, H, W))
    target_onehot = F.one_hot(target_indicies, num_classes=C)  # NHWC?
    input = F.softmax(torch.randn((N, C, H, W)), dim=1)

    return (input, target_indicies, target_onehot)


def xfail(val):
    """
    shorthand to mark xfail parameters
    """
    return pytest.param(val, marks=pytest.mark.xfail)


def generate_tensors():
    return [
        # binary classification
        _generate_tensors_classification(batch_size=64, num_classes=2),
        # classification
        _generate_tensors_classification(batch_size=64, num_classes=10),
        # segmentation
        xfail(_generate_tensors_segmentation(batch_size=64, num_classes=2, H=5, W=5)),
        xfail(_generate_tensors_segmentation(batch_size=64, num_classes=10, H=5, W=5))
    ]


@pytest.mark.parametrize('tensors', generate_tensors())
class TestSoftCrossEntropy:

    def test_infer_target_type(self, tensors):
        (input, target_indicies, target_onehot) = tensors
        assert loss._infer_target_type(input, target_indicies) == 'indicies'
        assert loss._infer_target_type(input, target_onehot) == 'one_hot'

    @pytest.mark.parametrize('reduction', ['mean', 'sum'])
    @pytest.mark.parametrize('use_weights', [xfail(True), False])
    def test_soft_cross_entropy(self, tensors, use_weights, reduction):
        (input, target_indicies, target_onehot) = tensors
        if use_weights:
            num_classes = target_onehot.shape[-1]
            weights = torch.randn(size=[num_classes])
        else:
            weights = None

        loss_args = dict(weight=weights, reduction=reduction)

        loss_indicies = loss.soft_cross_entropy(input, target_indicies, **loss_args)
        loss_onehot = loss.soft_cross_entropy(input, target_onehot, **loss_args)
        loss_reference = F.cross_entropy(input, target_indicies, **loss_args)

        torch.testing.assert_allclose(loss_indicies, loss_onehot)
        torch.testing.assert_allclose(loss_indicies, loss_reference)


@pytest.mark.parametrize('alpha', [0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('tensors', generate_tensors())
class TestLabelSmoothing:

    @staticmethod
    def reference_smooth_labels(targets, alpha):
        num_classes = targets.shape[-1]
        return targets * (1 - alpha) + alpha / num_classes

    def test_label_smoothing(self, tensors, alpha):
        (input, target_indicies, target_onehot) = tensors

        labels_onehot = label_smoothing.smooth_labels(input, target_onehot, alpha)
        labels_indicies = label_smoothing.smooth_labels(input, target_indicies, alpha)
        labels_ref = self.reference_smooth_labels(target_onehot, alpha)

        torch.testing.assert_allclose(labels_onehot, labels_ref)
        torch.testing.assert_allclose(labels_indicies, labels_ref)

    @pytest.mark.parametrize('target_type', ['onehot', 'indicies'])
    def test_label_smoothing_algorithm(self, tensors, alpha, target_type, dummy_logger, dummy_state):
        (outputs, target_indicies, target_onehot) = tensors

        target = target_indicies if target_type == 'indicies' else target_onehot

        algorithm = LabelSmoothingHparams(alpha=alpha).initialize_object()
        state = dummy_state
        state.batch = (torch.Tensor(), target)
        state.outputs = outputs

        # BEFORE_LOSS should smooth the labels
        algorithm.apply(Event.BEFORE_LOSS, state, dummy_logger)
        smoothed_reference = self.reference_smooth_labels(target_onehot, alpha)

        _, labels = state.batch
        torch.testing.assert_allclose(labels, smoothed_reference)

        # AFTER_LOSS should restore the original targets
        algorithm.apply(Event.AFTER_LOSS, state, dummy_logger)

        _, labels = state.batch
        torch.testing.assert_allclose(labels, target)


def test_label_smoothing_match():
    algorithm = LabelSmoothingHparams(alpha=0.1).initialize_object()
    assert algorithm.match(Event.BEFORE_LOSS, Mock())
    assert algorithm.match(Event.AFTER_LOSS, Mock())


@pytest.mark.run_long
@pytest.mark.timeout(90)
def test_label_smoothing_trains(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.algorithms = [LabelSmoothingHparams(alpha=0.1)]
    train_model(mosaic_trainer_hparams)
