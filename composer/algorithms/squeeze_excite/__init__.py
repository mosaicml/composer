# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExcite as SqueezeExcite
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExcite2d as SqueezeExcite2d
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExciteConv2d as SqueezeExciteConv2d
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExciteHparams as SqueezeExciteHparams
from composer.algorithms.squeeze_excite.squeeze_excite import apply_se as apply_se

_name = 'SqueezeExcite'
_class_name = 'SqueezeExcite'
_functional = 'apply_se'
_tldr = 'Replaces eligible layers with Squeeze-Excite layers'
_attribution = 'Hu et al, 2017'
_link = 'https://arxiv.org/abs/1709.01507'
_method_card = ''
