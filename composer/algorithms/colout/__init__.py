# Copyright 2021 MosaicML. All Rights Reserved.

"""Drops a fraction of the rows and columns of an input image. If the fraction of rows/columns dropped isn't too large,
this does not significantly alter the content of the image, but reduces its size and provides extra variability.

See the :doc:`Method Card </method_cards/colout>` for more details.
"""

from composer.algorithms.colout.colout import ColOut as ColOut
from composer.algorithms.colout.colout import ColOutTransform as ColOutTransform
from composer.algorithms.colout.colout import colout_batch as colout_batch

__all__ = ["ColOut", "ColOutTransform", "colout_batch"]
