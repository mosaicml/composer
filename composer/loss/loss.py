# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Custom loss functions."""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from composer.loss.utils import ensure_targets_one_hot, infer_target_type

__all__ = ["binary_cross_entropy_with_logits", "loss_registry", "soft_cross_entropy"]


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "sum",
    pos_weight: Optional[Tensor] = None,
    scale_by_batch_size: Optional[bool] = True,
) -> torch.Tensor:
    r"""Replacement for
    :class:`~torch.nn.functional.binary_cross_entropy_with_logits` that can handle class
    indices or one-hot labels.

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
        size_average (bool, optional): Deprecated (see `reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field ``size_average``
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see ``reduction``). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on `size_average`. When ``reduce`` is ``False``, returns a loss per
            batch element instead and ignores `size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: ``size_average``
            and ``reduce`` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override ``reduction``. Default:
            ``'sum'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.
        scale_by_batch_size (bool, optional): Whether to scale the loss by the batch size
            (i.e. input.shape[0]). Default: ``True``.
    """
    target = ensure_targets_one_hot(input, target)
    bce = F.binary_cross_entropy_with_logits(input, target, weight, size_average, reduce, reduction, pos_weight)
    if scale_by_batch_size:
        bce /= torch.tensor(input.shape[0])
    return bce


def soft_cross_entropy(input: Tensor,
                       target: Tensor,
                       weight: Optional[Tensor] = None,
                       size_average: Optional[bool] = None,
                       ignore_index: int = -100,
                       reduce: Optional[bool] = None,
                       reduction: str = 'mean'):
    r"""Drop-in replacement for :class:`~torch.nn.functional.cross_entropy` that can
     handle class indices or one-hot labels.
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
        size_average (bool, optional): Deprecated (see `reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field ``size_average``
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When ``size_average`` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            ``ignore_index`` is only applicable when the target contains class indices.
            Default: ``-100``
        reduce (bool, optional): Deprecated (see ``reduction``). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on `size_average`. When ``reduce`` is ``False``, returns a loss per
            batch element instead and ignores `size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: ``size_average``
            and ``reduce`` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override ``reduction``. Default: ``'mean'``
    This function will be obsolete with `this update <https://github.com/pytorch/pytorch/pull/61044>`_.
    """
    target_type = infer_target_type(input, target)

    if target_type == 'indices':
        return F.cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction)
    elif target_type == 'one_hot':
        assert reduction in ['sum', 'mean', 'none'], f"{reduction} reduction not supported."
        assert size_average is None, "size_average is deprecated"
        assert reduce is None, "reduce is deprecated"
        if ignore_index != -100:
            warnings.warn("ignore_index not supported when using dense labels. Ignoring targets with 0 probability.")
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
                raise ValueError("No targets have nonzero probability")
            if total_prob < num_examples:
                warnings.warn("Some targets have less than 1 total probability.")
            xentropy *= num_examples / total_prob

        return xentropy
    else:
        raise ValueError(f"Unrecognized target type {target_type}")


loss_registry = {
    "binary_cross_entropy_with_logits": binary_cross_entropy_with_logits,
    "soft_cross_entropy": soft_cross_entropy
}
