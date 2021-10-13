# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.layer_freezing.layer_freezing import LayerFreezing as LayerFreezing
from composer.algorithms.layer_freezing.layer_freezing import LayerFreezingHparams as LayerFreezingHparams
from composer.algorithms.layer_freezing.layer_freezing import freeze_layers as freeze_layers

_name = 'Layer Freezing'
_class_name = 'LayerFreezing'
_functional = 'freeze_layers'
_tldr = 'Progressively freezes layers during training.'
_attribution = 'Many (Raghu et al, 2017)'
_link = 'https://arxiv.org/abs/1706.05806'
_method_card = ''
