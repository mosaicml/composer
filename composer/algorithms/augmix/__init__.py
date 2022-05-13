# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""AugMix (`Hendrycks et al, 2020 <http://arxiv.org/abs/1912.02781>`_) creates multiple independent realizations of
sequences of image augmentations, applies each sequence with random intensity, and returns a convex combination of the
augmented images and the original image.

See the :doc:`Method Card </method_cards/augmix>` for more details.
"""

from composer.algorithms.augmix.augmix import AugmentAndMixTransform, AugMix, augmix_image

__all__ = ['AugmentAndMixTransform', 'AugMix', 'augmix_image']
