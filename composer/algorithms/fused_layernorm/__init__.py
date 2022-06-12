# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedLayerNorm
<https://nvidia.github.io/apex/layernorm.html>`_.

By fusing multiple kernel launches into one, this usually improves GPU utilization.

See the :doc:`Method Card </method_cards/fused_layernorm>` for more details.
"""

from composer.algorithms.fused_layernorm.fused_layernorm import FusedLayerNorm, apply_fused_layernorm

__all__ = ['FusedLayerNorm', 'apply_fused_layernorm']
