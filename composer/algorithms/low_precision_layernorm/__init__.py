# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces all instances of `torch.nn.LayerNorm` with a low precision `torch.nn.LayerNorm` (either float16 or bfloat16).
By default, torch.autocast always runs torch.nn.LayerNorm in float32, so this surgery forces a lower precision.
"""

from composer.algorithms.low_precision_layernorm.low_precision_layernorm import (LowPrecisionLayerNorm,
                                                                                 apply_low_precision_layernorm)

__all__ = ['LowPrecisionLayerNorm', 'apply_low_precision_layernorm']
