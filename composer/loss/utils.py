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
                           num_classes: Optional[int] = None) -> torch.Tensor:
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

        # Convert to one-hot tensor
        targets = _one_hot(targets, num_classes=num_classes, dim=1)
    return targets.float()


def check_for_index_targets(targets: torch.Tensor) -> bool:
    """Checks if a given set of targets are indices by looking at the type."""
    index_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    return targets.dtype in index_dtypes


def _one_hot(tensor: torch.Tensor, num_classes: int = -1, dim: int = -1) -> torch.Tensor:
    """Converts a tensor of index class labels to a tensor of one-hot class labels.

    Implementation is based on MONAI one-hot conversion function:
    `<https://github.com/Project-MONAI/MONAI/blob/b390b0956334325edc0e5000afb58e2be7cbe550/monai/networks/utils.py#L49>`_.

    Args:
        tensor (torch.Tensor): Tensor containing index class labels.
        num_classes (int): Size of the class dimension for the output one-hot tensor. If set to -1,
            the number of classes will be inferred to be one greater than the largest value in ``tensor``.
        dim (int): Location of the new class dimension of size ``num_classes``.


    Returns:
        torch.Tensor: One-hot class labels i.e. the same shape as ``tensor`` except with an
            extra dimension of size ``num_classes`` inserted after the first dimension
    """
    if not check_for_index_targets(tensor):
        raise ValueError(f'tensor must be integer type, current type: {tensor.dtype}')

    max_index = tensor.max() + 1
    if num_classes == -1:
        num_classes = int(max_index)

    if num_classes < max_index:
        raise ValueError(f'num_classes must be greater than or equal to tensor.max() + 1: {num_classes} < {max_index}')

    # Remove negative indices
    neg_indices = tensor.min() < 0
    if neg_indices:
        warnings.warn('Negative label indices are being ignored in conversion to one-hot labels')
        tensor = tensor.clone().long()
        tensor[tensor < 0] = num_classes
        num_classes += 1  # Add extra class for negative indices

    # Assume class dimension is inserted after the first dimension
    tensor = tensor.unsqueeze(dim)
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = num_classes

    # Convert to one-hot
    one_hot_tensor = torch.zeros(size=tensor_shape, dtype=tensor.dtype, device=tensor.device)
    one_hot_tensor.scatter_(dim=dim, index=tensor, value=1)

    # Remove negative indices
    if neg_indices:
        one_hot_tensor = one_hot_tensor[:, 0:-1]

    return one_hot_tensor
