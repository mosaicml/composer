# Copyright 2021 MosaicML. All Rights Reserved.

"""Drops a fraction of the rows and columns of an input image. If the fraction of
rows/columns dropped isn't too large, this does not significantly alter the content
of the image, but reduces its size and provides extra variability. See the
:doc:`Method Card </method_cards/col_out>` for more details.
"""

from composer.algorithms.colout.colout import ColOut as ColOut
from composer.algorithms.colout.colout import colout_batch as colout_batch
from composer.algorithms.colout.colout import colout_image as colout_image

_name = 'ColumnOut'
_class_name = 'ColOut'
_functional = 'colout_batch'
_tldr = 'Removes columns and rows from the image for augmentation and efficiency.'
_attribution = 'MosaicML'
_link = ''
_method_card = 'docs/source/method_cards/col_out.md'

__all__ = ["ColOut", "ColOutTransform", "colout_image", "colout_batch"]