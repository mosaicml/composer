# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.blurpool.blurpool import BlurPool as BlurPool
from composer.algorithms.blurpool.blurpool import BlurPoolHparams as BlurPoolHparams
from composer.algorithms.blurpool.blurpool import apply_blurpool as apply_blurpool
from composer.algorithms.blurpool.blurpool_layers import BlurConv2d as BlurConv2d
from composer.algorithms.blurpool.blurpool_layers import BlurMaxPool2d as BlurMaxPool2d
from composer.algorithms.blurpool.blurpool_layers import BlurPool2d as BlurPool2d
from composer.algorithms.blurpool.blurpool_layers import blur_2d as blur_2d

_name = 'BlurPool'
_class_name = 'BlurPool'
_functional = 'apply_blurpool'
_tldr = ''
_attribution = '(Zhang, 2019)'
_link = 'https://arxiv.org/abs/1904.11486'
_method_card = ''
