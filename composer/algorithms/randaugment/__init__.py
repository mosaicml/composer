# Copyright 2021 MosaicML. All Rights Reserved.

"""RandAugment (`Cubuk et al., 2019
<https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf>`_)
randomly applies a sequence of image data augmentations to a single image. See the
:doc:`Method Card </method_cards/rand_augment>` for more details.
"""

from composer.algorithms.randaugment.randaugment import RandAugment
from composer.algorithms.randaugment.randaugment import RandAugmentTransform
from composer.algorithms.randaugment.randaugment import randaugment_image

__all__ = ['RandAugment', 'RandAugmentTransform', 'randaugment_image']

