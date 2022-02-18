# Copyright 2021 MosaicML. All Rights Reserved.

"""`CutMix <https://arxiv.org/abs/1905.04899>`_ trains the network on non-overlapping combinations of pairs of examples
and iterpolated targets rather than individual examples and targets.

This is done by taking a non-overlapping combination of a given batch X with a
randomly permuted copy of X.

See the :doc:`Method Card </method_cards/cutmix>` for more details.
"""

from composer.algorithms.cutmix.cutmix import CutMix as CutMix
from composer.algorithms.cutmix.cutmix import cutmix_batch as cutmix_batch

__all__ = ["CutMix", "cutmix_batch"]
