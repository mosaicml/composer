# Copyright 2021 MosaicML. All Rights Reserved.

"""`CutMix <https://arxiv.org/abs/1905.04899>`_ trains the network on non-overlapping combinations of pairs of
examples and iterpolated targets rather than individual examples and targets.

This is done by taking a non-overlapping combination of a given batch X with a
randomly permuted copy of X.
"""

from composer.algorithms.cutmix.cutmix import CutMix as CutMix
from composer.algorithms.cutmix.cutmix import cutmix_batch as cutmix_batch

_name = 'CutMix'
_class_name = 'CutMix'
_functional = 'cutmix_batch'
_tldr = 'Combines pairs of examples in non-overlapping regions and mixes labels'
_attribution = '(Yun et al, 2019)'
_link = 'https://arxiv.org/abs/1905.04899'
_method_card = 'docs/source/method_cards/cutmix.md'

__all__ = ["CutMix", "cutmix_batch"]