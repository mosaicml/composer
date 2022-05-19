# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Randomly applies a sequence of image data augmentations
(`Cubuk et al, 2019 <https://arxiv.org/abs/1909.13719>`_) to an image. See
:class:`.RandAugment` or the :doc:`Method Card
</method_cards/randaugment>` for details.
"""

from composer.algorithms.randaugment.randaugment import RandAugment, RandAugmentTransform, randaugment_image

__all__ = ['RandAugment', 'RandAugmentTransform', 'randaugment_image']
