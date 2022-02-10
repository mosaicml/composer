# Copyright 2021 MosaicML. All Rights Reserved.

"""AugMix (`Hendrycks et al., 2020 <http://arxiv.org/abs/1912.02781>`_)
creates multiple independent realizations of sequences of image augmentations, applies
each sequence with random intensity, and returns a convex combination of the augmented
images and the original image. See the :doc:`Method Card </method_cards/aug_mix>` for more
details.
"""

from composer.algorithms.augmix.augmix import AugmentAndMixTransform as AugmentAndMixTransform
from composer.algorithms.augmix.augmix import AugMix as AugMix
from composer.algorithms.augmix.augmix import augmix_image as augmix_image

_name = 'AugMix'
_class_name = 'AugMix'
_functional = 'augmix_image'
_tldr = 'Image-perserving data augmentations'
_attribution = '(Hendrycks et al, 2020)'
_link = 'http://arxiv.org/abs/1912.02781'
_method_card = ''

__all__ = ['AugMix', 'augmix_image']
