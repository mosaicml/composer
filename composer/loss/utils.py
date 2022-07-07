# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Loss-related utilities."""

from __future__ import annotations

import warnings
from typing import Optional

import torch

__all__ = ['infer_target_type', 'ensure_targets_one_hot', 'check_for_index_targets']


def infer_target_type(input: torch.Tensor, targets: torch.Tensor) -> str:
    """Infers whether the target is in indices format or one_hot format.

    Example indices format: [1, 4, 7] Example one_hot format [[0, 1, 0], [1, 0, 0], ...]
    """
    if input.shape == targets.shape:
        return 'one_hot'
    elif input.ndim == targets.ndim + 1:
        return 'indices'
    else:
        raise RuntimeError(f'Unable to infer indices or one_hot. Targets has shape {targets.shape}'
                           f' and the inputs to cross entropy has shape {input.shape}. For one_hot, '
                           'expect targets.shape == inputs.shape. For indices, expect '
                           'inputs.ndim == targets.ndim + 1')


def ensure_targets_one_hot(input: torch.Tensor,
                           targets: torch.Tensor,
                           num_classes: Optional[float] = None) -> torch.Tensor:
    r"""Ensures that the targets are in a one-hot format rather than an index format.

    Args:
        input (torch.Tensor): :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        targets (torch.Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        num_classes (int, optional): Number of classes. If not specified, this will be inferred
            from input. Default: ``None``
    """

    if infer_target_type(input, targets) == 'indices':
        # If the number of classes isn't specified, attempt to infer it from the input
        if num_classes is None:
            num_classes = input.shape[1]
        if targets.min() < 0:
            warnings.warn('Negative label indices are being ignored in conversion to one-hot labels')

            # Add new dimension for the class vectors
            class_dim = 1
            targets = targets.clone().unsqueeze(class_dim).long()

            # Map all negative indicies to a class to drop.
            targets[targets < 0] = num_classes

            # Create one-hot labels
            input_shape = list(input.shape)
            input_shape[class_dim] += 1  # Add additional class for negative indicies
            one_hot_targets = torch.zeros(size=input_shape, dtype=input.dtype, device=input.device)
            one_hot_targets.scatter_(dim=1, index=targets, value=1)

            # Drop any negative indices.
            one_hot_targets = one_hot_targets[:, 0:-1]
        else:
            one_hot_targets = torch.zeros_like(input)
            one_hot_targets.scatter_(dim=1, index=targets, value=1)
    else:
        one_hot_targets = targets

    one_hot_targets = one_hot_targets.float()
    return one_hot_targets


def check_for_index_targets(targets: torch.Tensor) -> bool:
    """Checks if a given set of targets are indices by looking at the type."""
    index_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    return targets.dtype in index_dtypes
