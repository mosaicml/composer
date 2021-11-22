# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.cutmix.cutmix import CutMix as CutMix
from composer.algorithms.cutmix.cutmix import CutMixHparams as CutMixHparams
from composer.algorithms.cutmix.cutmix import cutmix as cutmix

_name = 'CutMix'
_class_name = 'CutMix'
_functional = 'cutmix'
_tldr = 'Combines pairs of examples in non-overlapping regions and mixes labels'
_attribution = '(Yun et al, 2019)'
_link = 'https://arxiv.org/abs/1905.04899'
_method_card = ''
