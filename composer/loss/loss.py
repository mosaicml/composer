# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Custom loss functions."""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from composer.loss.utils import ensure_targets_one_hot, infer_target_type

__all__ = ['binary_cross_entropy_with_logits', 'loss_registry', 'soft_cross_entropy']


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = 'sum',
    pos_weight: Optional[Tensor] = None,
    scale_by_batch_size: Optional[bool] = True,
) -> torch.Tensor:
    r"""Replacement for :class:`~F.binary_cross_entropy_with_logits` that handles class indices or one-hot labels.

    :class:`~torch.nn.functional.binary_cross_entropy_with_logits` with ``reduction =
    'mean'` will typically result in very small loss values (on the order of 1e-3), which
    necessitates scaling the learning rate by 1e3 to allow the model to learn. This
    implementation avoids this by using ``reduction = sum`` and scaling the loss inversely
    proportionally to the batch size.

    Args:
        input (torch.Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        target (torch.Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        weight (torch.Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`. Default: ``None``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default:
            ``'sum'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.
        scale_by_batch_size (bool, optional): Whether to scale the loss by the batch size
            (i.e. input.shape[0]). Default: ``True``.
    """
    target = ensure_targets_one_hot(input, target)
    bce = F.binary_cross_entropy_with_logits(
        input=input,
        target=target,
        weight=weight,
        reduction=reduction,
        pos_weight=pos_weight,
    )
    if scale_by_batch_size:
        bce /= torch.tensor(input.shape[0])
    return bce


def soft_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
):
    r"""Drop-in replacement for :class:`~.F.cross_entropy` that handles class indices or one-hot labels.

    .. note::

        This function will be obsolete with `this update <https://github.com/pytorch/pytorch/pull/61044>`_.

    Args:
        input (torch.Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        target (torch.Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        weight (torch.Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`. Default: ``None``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When ``size_average`` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            ``ignore_index`` is only applicable when the target contains class indices.
            Default: ``-100``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """
    target_type = infer_target_type(input, target)

    if target_type == 'indices':
        return F.cross_entropy(
            input=input,
            target=target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )
    elif target_type == 'one_hot':
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(f'{reduction} reduction not supported.')
        if ignore_index != -100:
            warnings.warn('ignore_index not supported when using dense labels. Ignoring targets with 0 probability.')
        xentropy = -(target * F.log_softmax(input, dim=1))

        if weight is not None:
            # Ugly dimension shuffle to make multiplication work.
            xentropy = torch.movedim(xentropy, 1, -1)
            xentropy *= weight  # PyTorch doesn't normalize weights
            xentropy = torch.movedim(xentropy, -1, 1)

        xentropy = xentropy.sum(dim=1)
        num_examples = torch.numel(xentropy)

        if reduction == 'sum':
            xentropy = xentropy.sum()
        elif reduction == 'mean':
            xentropy = xentropy.mean()
            # Re-weight loss to account for examples with less than 1 total probability (ignored examples)
            total_prob = target.sum()
            if total_prob <= 0:
                raise ValueError('No targets have nonzero probability')
            if total_prob < num_examples:
                warnings.warn('Some targets have less than 1 total probability.')
            xentropy *= num_examples / total_prob

        return xentropy
    else:
        raise ValueError(f'Unrecognized target type {target_type}')


class DiceLoss(_Loss):
    """Criterion that computes the dice loss between input and target.

    The implementation is derived from MONAI: `<https://docs.monai.io/en/stable/losses.html#diceloss>`_.
    For more information about the dice loss see the original paper on dice loss:
    `<https://arxiv.org/abs/1606.04797>`_.

    Args:
        sigmoid (bool): If true, apply a sigmoid function to the input. Default: ``False``
        softmax (bool): If true, apply a softmax function to the input. Default: ``False``
        squared_pred (bool): If true, square the inputs and targets when calculating the
            class unions. Default: ``False``
        jaccard (bool): If true, compute the jaccard index (soft IoU) instead of dice.
            Default: ``False``
        batch (bool): If true, sum the intersection and union areas over the batch
            dimension before dividing the two quantities. If false, a dice loss value is
            computed independently for each sample in the batch before the reduction.
        ignore_absent_classes (bool): If true, remove classes that are not present in
            the target from the loss calculation. Classes not present in the target do
            not contribute to the gradient, but can decrease the weight of present classes,
            slowing optimization. This should have no effect if all classes are present in
            each sample. Default: ``'False'``
        reduction (str): Specifies the reduction to apply to the output: ``'none'`` |
            ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be appied, ``'mean'``:
            the weighted mean of the output is taken, ``'sum'``: the output will be summed.
            Default: ``'mean'``

    """

    def __init__(
        self,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        batch: bool = False,
        ignore_absent_classes: bool = False,
        reduction: str = 'mean',
    ):
        super().__init__(reduction=reduction)
        if sigmoid and softmax:
            raise ValueError('Both sigmoid and softmax should not be true.')
        if not reduction in ['none', 'mean', 'sum']:
            raise ValueError(f'reduction was {reduction}, but must be one of ["none", "mean", "sum"]')

        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.reduction = reduction
        self.batch = batch
        self.ignore_absent_classes = ignore_absent_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # If target is not one-hot, convert to one-hot
        target = ensure_targets_one_hot(input, target)

        # Get mask of pixels with a target
        target_mask = target.sum(dim=1, keepdim=True) != 0

        if input.shape != target.shape:
            raise AssertionError(f'ground truth has different shape ({target.shape}) from input ({input.shape})')

        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn('single channel prediction, `softmax=True` ignored.')
            else:
                input = torch.softmax(input, 1)

        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        # Zero out pixels which do not have a target
        input = target_mask * input

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        union = ground_o + pred_o

        if self.jaccard:
            union = 2.0 * (union - intersection)

        epsilon = 1e-5
        ious = 1.0 - (2.0 * intersection + epsilon) / (union + epsilon)

        if self.ignore_absent_classes:
            if self.batch:
                ious = ious[ground_o > 0]
            else:
                ious = ious[:, (ground_o.sum(dim=0) > 0)]

        if self.reduction == 'mean':
            iou = torch.mean(ious)  # the batch and channel average
        elif self.reduction == 'sum':
            iou = torch.sum(ious)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            iou = ious
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return iou


loss_registry = {
    'binary_cross_entropy_with_logits': binary_cross_entropy_with_logits,
    'soft_cross_entropy': soft_cross_entropy,
}
