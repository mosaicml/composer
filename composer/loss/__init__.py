# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of custom loss functions and loss function related utilities."""

from composer.loss.loss import DiceLoss, binary_cross_entropy_with_logits, loss_registry, soft_cross_entropy

__all__ = [
    'DiceLoss',
    'binary_cross_entropy_with_logits',
    'loss_registry',
    'soft_cross_entropy',
]
