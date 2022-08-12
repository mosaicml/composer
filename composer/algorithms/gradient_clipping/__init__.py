# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Clips all gradients in a model based on their values, their norms,
and their parameters' norms.

See the :doc:`Method Card </method_cards/gradient_clipping>` for more details.
"""

from composer.algorithms.gradient_clipping.gradient_clipping import GradientClipping, apply_gradient_clipping

__all__ = ['GradientClipping', 'apply_gradient_clipping']
