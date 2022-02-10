# Copyright 2021 MosaicML. All Rights Reserved.

"""RandAugment randomly applies a sequence of image data augmentations to a single image
(`Cubuk et al. 2019
<https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf>`_).
"""

from composer.algorithms.randaugment.randaugment import RandAugment as RandAugment
from composer.algorithms.randaugment.randaugment import RandAugmentHparams as RandAugmentHparams
from composer.algorithms.randaugment.randaugment import RandAugmentTransform as RandAugmentTransform
from composer.algorithms.randaugment.randaugment import randaugment_image as randaugment_image

_name = 'RandAugment'
_class_name = 'RandAugment'
_functional = 'randaugment_image'
_tldr = 'Applies a series of random augmentations'
_attribution = '(Cubuk et al, 2020)'
_link = 'https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html'
_method_card = ''

__all__ = ['RandAugment', 'RandAugmentHparams', "RandAugmentTransform", 'randaugment_image']
