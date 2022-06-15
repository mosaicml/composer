# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`BlurPool <http://proceedings.mlr.press/v97/zhang19a.html>`_ adds anti-aliasing filters to convolutional layers to
increase accuracy and invariance to small shifts in the input.

See :class:`~composer.algorithms.BlurPool` or the :doc:`Method Card </method_cards/blurpool>` for details.
"""

from composer.algorithms.blurpool.blurpool import BlurPool as BlurPool
from composer.algorithms.blurpool.blurpool import apply_blurpool as apply_blurpool
from composer.algorithms.blurpool.blurpool_layers import BlurConv2d as BlurConv2d
from composer.algorithms.blurpool.blurpool_layers import BlurMaxPool2d as BlurMaxPool2d
from composer.algorithms.blurpool.blurpool_layers import BlurPool2d as BlurPool2d
from composer.algorithms.blurpool.blurpool_layers import blur_2d as blur_2d
from composer.algorithms.blurpool.blurpool_layers import blurmax_pool2d as blurmax_pool2d

__all__ = [
    'BlurPool',
    'apply_blurpool',
    'BlurConv2d',
    'BlurMaxPool2d',
    'BlurPool2d',
    'blur_2d',
    'blurmax_pool2d',
]
