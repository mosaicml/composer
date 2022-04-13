import numpy as np
import pytest
import torch
from torch.nn import functional as F

from composer.loss import soft_cross_entropy
from composer.loss.utils import ensure_targets_one_hot, infer_target_type


def fake_input_target_pairs(input_shape):
    num_classes = input_shape[1]
    reduced_input_shape = list(input_shape)
    reduced_input_shape.pop(1)

    input = torch.randn(input_shape)
    targets_idx = torch.randint(low=-1, high=num_classes, size=reduced_input_shape)
    targets_one_hot = torch.zeros_like(input)
    for i, value in np.ndenumerate(targets_idx):
        i_expanded = list(i)
        if value >= 0:
            i_expanded.insert(1, value)
            targets_one_hot[tuple(i_expanded)] = 1.0
    return input, targets_idx, targets_one_hot


def xfail(val):
    """shorthand to mark xfail parameters."""
    return pytest.param(val, marks=pytest.mark.xfail)


def generate_tensors():
    return [
        # Binary classification
        fake_input_target_pairs((64, 2)),
        # Classification
        fake_input_target_pairs((64, 10)),
        # Segmentation
        fake_input_target_pairs((64, 2, 5, 5)),
        fake_input_target_pairs((64, 10, 5, 5)),
        # 3D inputs
        fake_input_target_pairs((64, 2, 5, 7, 11)),
        fake_input_target_pairs((64, 10, 5, 7, 11))
    ]


@pytest.mark.parametrize('tensors', generate_tensors())
def test_ensure_targets_one_hot(tensors):
    input, targets_idx, targets_one_hot = tensors
    targets_one_hot_test = ensure_targets_one_hot(input, targets_idx)
    torch.testing.assert_allclose(targets_one_hot, targets_one_hot_test)


@pytest.mark.parametrize('tensors', generate_tensors())
class TestSoftCrossEntropy:

    def test_infer_target_type(self, tensors):
        input, target_indices, target_onehot = tensors
        assert infer_target_type(input, target_indices) == 'indices'
        assert infer_target_type(input, target_onehot) == 'one_hot'

    @pytest.mark.parametrize('reduction', ['mean', 'sum'])
    @pytest.mark.parametrize('use_weights', [xfail(True), False])
    def test_soft_cross_entropy(self, tensors, use_weights, reduction):
        input, target_indices, target_onehot = tensors
        if use_weights:
            num_classes = target_onehot.shape[1]
            weights = torch.rand(size=[num_classes])
        else:
            weights = None

        loss_args = dict(ignore_index=-1, weight=weights, reduction=reduction)
        loss_indices = soft_cross_entropy(input, target_indices, **loss_args)
        loss_onehot = soft_cross_entropy(input, target_onehot, **loss_args)
        loss_reference = F.cross_entropy(input, target_indices, **loss_args)

        torch.testing.assert_allclose(loss_indices, loss_onehot)
        torch.testing.assert_allclose(loss_indices, loss_reference)
