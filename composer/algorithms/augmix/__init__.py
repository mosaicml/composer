# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.augmix.augmix import AugmentAndMixTransform as AugmentAndMixTransform
from composer.algorithms.augmix.augmix import AugMix as AugMix
from composer.algorithms.augmix.augmix import AugMixHparams as AugMixHparams
from composer.algorithms.augmix.augmix import augment_and_mix as augment_and_mix

_name = 'AugMix'
_class_name = 'AugMix'
_functional = 'augment_and_mix'
_tldr = 'Image-perserving data augmentations'
_attribution = '(Hendrycks et al, 2020)'
_link = 'http://arxiv.org/abs/1912.02781'
_method_card = ''
