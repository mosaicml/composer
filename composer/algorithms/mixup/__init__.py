# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.mixup.mixup import MixUp as MixUp
from composer.algorithms.mixup.mixup import MixUpHparams as MixUpHparams
from composer.algorithms.mixup.mixup import mixup_batch as mixup_batch

_name = 'MixUp'
_class_name = 'MixUp'
_functional = 'mixup_batch'
_tldr = 'Blends pairs of examples and labels'
_attribution = '(Zhang et al, 2017)'
_link = 'https://arxiv.org/abs/1710.09412'
_method_card = ''
