# Copyright 2022 MosaicML. All Rights Reserved.

"""`DropBlock <https://arxiv.org/abs/1810.12890>`_ is a form of structured dropout, where units in a contiguous region
of a feature map are dropped together.

See the :doc:`Method Card </method_cards/dropblock>` for more details.
"""

from composer.algorithms.dropblock.dropblock import DropBlock as DropBlock
from composer.algorithms.dropblock.dropblock import dropblock_batch as dropblock_batch

__all__ = ["DropBlock", "dropblock_batch"]
