# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces the Linear layers in the feed-forward network with `Gated Linear Units <https://arxiv.org/abs/2002.05202>`_.

This leads to improved convergence with a slight drop in throughput. Using no bias terms in the GLU is highly recommended.

See the :doc:`Method Card </method_cards/gated_linear_units>` for more details.
"""

from composer.algorithms.gated_linear_units.gated_linear_units import GatedLinearUnits, apply_gated_linear_units

__all__ = ['GatedLinearUnits', 'apply_gated_linear_units']
