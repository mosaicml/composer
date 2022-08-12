# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`Cutout <https://arxiv.org/abs/1708.04552>`_ is a data augmentation technique that works by masking out one or more
square regions of an input image.

See the :doc:`Method Card </method_cards/cutout>` for more details.
"""

from composer.algorithms.cutout.cutout import CutOut as CutOut
from composer.algorithms.cutout.cutout import cutout_batch as cutout_batch

__all__ = ['CutOut', 'cutout_batch']
