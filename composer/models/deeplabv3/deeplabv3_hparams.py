from dataclasses import dataclass
from typing import List

import yahp as hp

from composer.models.model_hparams import Initializer, ModelHparams


@dataclass
class DeepLabv3Hparams(ModelHparams):
    backbone_arch: str = hp.optional("The backbone architecture to use. Must be either ['resnet50', resnet101']",
                                     default='resnet101')
    is_backbone_pretrained: bool = hp.optional("Whether or not to use a pretrained backbone", default=True)
    sync_bn: bool = hp.optional("Whether or not to sync the batch statistics across devices", default=False)

    def initialize_object(self):
        from composer.models.deeplabv3.deeplabv3 import MosaicDeepLabV3
        return MosaicDeepLabV3(self)
