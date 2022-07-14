# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from composer.algorithms import LabelSmoothing, label_smoothing
from composer.core import Event


def _generate_tensors_classification(batch_size: int, num_classes: int):
    """Helper functions to generate input, target pairs for image classification (1d indices)"""
    N = batch_size
    C = num_classes

    target_indices = torch.randint(0, C, [N])
    target_onehot = F.one_hot(target_indices, num_classes=C)
    input = F.softmax(torch.randn((N, C)), dim=1)

    return (input, target_indices, target_onehot)


def _generate_tensors_segmentation(batch_size: int, num_classes: int, H: int, W: int):
    """Helper functions to generate input, target pairs for image segmentation (2d indices)"""
    N = batch_size
    C = num_classes

    target_indices = torch.randint(0, C, (N, H, W))
    target_onehot = F.one_hot(target_indices, num_classes=C)  # NHWC
    target_onehot = torch.movedim(target_onehot, -1, 1).contiguous()  # NCHW
    input = F.softmax(torch.randn((N, C, H, W)), dim=1)

    return (input, target_indices, target_onehot)


def xfail(val):
    """shorthand to mark xfail parameters."""
    return pytest.param(val, marks=pytest.mark.xfail)


def generate_tensors():
    return [
        # binary classification
        _generate_tensors_classification(batch_size=64, num_classes=2),
        # classification
        _generate_tensors_classification(batch_size=64, num_classes=10),
        # segmentation
        _generate_tensors_segmentation(batch_size=64, num_classes=2, H=5, W=5),
        _generate_tensors_segmentation(batch_size=64, num_classes=10, H=5, W=5)
    ]


@pytest.mark.parametrize('smoothing', [0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('tensors', generate_tensors())
class TestLabelSmoothing:

    @staticmethod
    def reference_smooth_labels(targets, smoothing):
        num_classes = targets.shape[1]
        return targets * (1 - smoothing) + smoothing / num_classes

    def test_label_smoothing(self, tensors, smoothing):
        (input, target_indices, target_onehot) = tensors

        labels_onehot = label_smoothing.smooth_labels(input, target_onehot, smoothing)
        labels_indices = label_smoothing.smooth_labels(input, target_indices, smoothing)
        labels_ref = self.reference_smooth_labels(target_onehot, smoothing)

        torch.testing.assert_close(labels_onehot, labels_ref)
        torch.testing.assert_close(labels_indices, labels_ref)

    @pytest.mark.parametrize('target_type', ['onehot', 'indices'])
    def test_label_smoothing_algorithm(self, tensors, smoothing, target_type, empty_logger, minimal_state):
        (outputs, target_indices, target_onehot) = tensors

        target = target_indices if target_type == 'indices' else target_onehot

        algorithm = LabelSmoothing(smoothing=smoothing)
        state = minimal_state
        state.batch = (torch.Tensor(), target)
        state.outputs = outputs

        # BEFORE_LOSS should smooth the labels
        algorithm.apply(Event.BEFORE_LOSS, state, empty_logger)
        smoothed_reference = self.reference_smooth_labels(target_onehot, smoothing)

        _, labels = state.batch
        torch.testing.assert_close(labels, smoothed_reference)

        # AFTER_LOSS should restore the original targets
        algorithm.apply(Event.AFTER_LOSS, state, empty_logger)

        _, labels = state.batch
        torch.testing.assert_close(labels, target)
