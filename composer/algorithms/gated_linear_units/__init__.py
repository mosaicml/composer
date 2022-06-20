# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.GatedLinearUnits
<https://nvidia.github.io/apex/layernorm.html>`_.
TODO (Moin): update this.

By fusing multiple kernel launches into one, this usually improves GPU utilization.

See the :doc:`Method Card </method_cards/gated_linear_units>` for more details.
"""

from composer.algorithms.gated_linear_units.gated_linear_units import GatedLinearUnits, apply_gated_linear_units

__all__ = ['GatedLinearUnits', 'apply_gated_linear_units']
