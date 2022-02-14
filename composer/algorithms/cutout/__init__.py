# Copyright 2021 MosaicML. All Rights Reserved.

"""`Cutout <https://arxiv.org/abs/1708.04552>`_ is a data augmentation technique that works by masking out one or
more square regions of an input image. See the :doc:`Method Card </method_cards/cut_out>` for more details."""

from composer.algorithms.cutout.cutout import CutOut as CutOut
from composer.algorithms.cutout.cutout import cutout_batch as cutout_batch

_name = 'CutOut'
_class_name = 'CutOut'
_functional = 'cutout_batch'
_tldr = 'Randomly erases rectangular blocks from the image.'
_attribution = '(DeVries et al, 2017)'
_link = 'https://arxiv.org/abs/1708.04552'
_method_card = 'docs/source/method_cards/cut_out.md'

__all__ = ["CutOut", "cutout_batch"]
