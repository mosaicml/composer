# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of custom loss functions and loss function related utilities."""

from composer.loss.loss import DiceLoss as DiceLoss
from composer.loss.loss import binary_cross_entropy_with_logits as binary_cross_entropy_with_logits
from composer.loss.loss import loss_registry as loss_registry
from composer.loss.loss import soft_cross_entropy as soft_cross_entropy
