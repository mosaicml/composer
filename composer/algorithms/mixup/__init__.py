# Copyright 2021 MosaicML. All Rights Reserved.

"""Create new samples using convex combinations of pairs of samples. This is done by taking a
convex combination of x with a randomly permuted copy of x.
See the :doc:`Method Card </method_cards/mix_up>` for more details.
"""

from composer.algorithms.mixup.mixup import MixUp as MixUp
from composer.algorithms.mixup.mixup import mixup_batch as mixup_batch

_name = 'MixUp'
_class_name = 'MixUp'
_functional = 'mixup_batch'
_tldr = 'Blends pairs of examples and labels'
_attribution = '(Zhang et al, 2017)'
_link = 'https://arxiv.org/abs/1710.09412'
_method_card = 'docs/source/method_cards/mix_up.md'

__all__ = ["MixUp", "mixup_batch"]
