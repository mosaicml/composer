# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from composer.loss import DiceLoss, soft_cross_entropy
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


@pytest.mark.filterwarnings(
    r'ignore:Negative label indices are being ignored in conversion to one-hot labels:UserWarning')
@pytest.mark.parametrize('tensors', generate_tensors())
def test_ensure_targets_one_hot(tensors):
    input, targets_idx, targets_one_hot = tensors
    targets_one_hot_test = ensure_targets_one_hot(input, targets_idx)
    torch.testing.assert_close(targets_one_hot, targets_one_hot_test, check_stride=False)


@pytest.mark.parametrize('tensors', generate_tensors())
class TestSoftCrossEntropy:

    def test_infer_target_type(self, tensors):
        input, target_indices, target_onehot = tensors
        assert infer_target_type(input, target_indices) == 'indices'
        assert infer_target_type(input, target_onehot) == 'one_hot'

    @pytest.mark.parametrize('reduction', ['mean', 'sum'])
    @pytest.mark.parametrize('use_weights', [xfail(True), False])
    # TODO(Cory): Remove this filterwarning
    @pytest.mark.filterwarnings(r'ignore:Some targets have less than 1 total probability:UserWarning')
    def test_soft_cross_entropy(self, tensors, use_weights, reduction):
        input, target_indices, target_onehot = tensors
        if use_weights:
            num_classes = target_onehot.shape[1]
            weights = torch.rand(size=[num_classes])
        else:
            weights = None

        loss_indices = soft_cross_entropy(input, target_indices, weight=weights, reduction=reduction, ignore_index=-1)
        loss_onehot = soft_cross_entropy(input, target_onehot, weight=weights, reduction=reduction)
        loss_reference = F.cross_entropy(input, target_indices, weight=weights, reduction=reduction, ignore_index=-1)

        torch.testing.assert_close(loss_indices, loss_onehot)
        torch.testing.assert_close(loss_indices, loss_reference)


class TestDiceLoss:

    @pytest.fixture()
    def target(self):
        target = torch.tensor([[[-1], [0], [1], [2]]]).repeat(1, 1, 4)
        target = torch.cat([target, target[:, [1, 2, 3, 0]], target[:, [2, 3, 0, 1]], target[:, [3, 0, 1, 2]]], dim=0)
        return target

    @pytest.fixture()
    def correct_input(self, target):
        input = target.clone()
        input[input == -1] = 1  # replace negative label with class prediction
        input = F.one_hot(input)
        input = torch.movedim(input, 3, 1)
        return input

    @pytest.fixture()
    def incorrect_input(self, correct_input):
        return correct_input[[3, 2, 1, 0]]

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('squared_pred', [True, False])
    @pytest.mark.parametrize('jaccard', [True, False])
    @pytest.mark.parametrize('batch', [True, False])
    @pytest.mark.parametrize('ignore_absent_classes', [True, False])
    @pytest.mark.parametrize('reduction', ['mean', 'sum'])
    def test_correct_prediction(self, correct_input: torch.Tensor, target: torch.Tensor, squared_pred: bool,
                                jaccard: bool, batch: bool, ignore_absent_classes: bool, reduction: str):
        dice_loss = DiceLoss(squared_pred=squared_pred,
                             jaccard=jaccard,
                             batch=batch,
                             ignore_absent_classes=ignore_absent_classes,
                             reduction=reduction)

        loss = dice_loss(correct_input, target)
        assert loss == 0.0

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('squared_pred', [True, False])
    @pytest.mark.parametrize('jaccard', [True, False])
    @pytest.mark.parametrize('batch', [True, False])
    @pytest.mark.parametrize('ignore_absent_classes', [True, False])
    def test_incorrect_prediction(self, incorrect_input: torch.Tensor, target: torch.Tensor, squared_pred: bool,
                                  jaccard: bool, batch: bool, ignore_absent_classes: bool):
        dice_loss = DiceLoss(squared_pred=squared_pred,
                             jaccard=jaccard,
                             batch=batch,
                             ignore_absent_classes=ignore_absent_classes)
        loss = dice_loss(incorrect_input, target)
        torch.testing.assert_close(loss, torch.tensor(1.0))
