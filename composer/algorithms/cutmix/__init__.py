# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`CutMix <https://arxiv.org/abs/1905.04899>`_ trains the network on non-overlapping combinations of pairs of examples
and iterpolated targets rather than individual examples and targets.

This is done by taking a non-overlapping combination of a given batch X with a
randomly permuted copy of X.

See the :doc:`Method Card </method_cards/cutmix>` for more details.
"""

from composer.algorithms.cutmix.cutmix import CutMix as CutMix
from composer.algorithms.cutmix.cutmix import cutmix_batch as cutmix_batch

__all__ = ['CutMix', 'cutmix_batch']
