# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces all instances of :class:`torch.nn.GroupNorm` with a low precision :class:`torch.nn.GroupNorm` (either float16 or bfloat16).
By default, torch.autocast always runs torch.nn.GroupNorm in float32, so this surgery forces a lower precision.
"""

from composer.algorithms.low_precision_groupnorm.low_precision_groupnorm import (
    LowPrecisionGroupNorm,
    apply_low_precision_groupnorm,
)

__all__ = ['LowPrecisionGroupNorm', 'apply_low_precision_groupnorm']
