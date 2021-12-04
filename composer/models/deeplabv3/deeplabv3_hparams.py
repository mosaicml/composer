from dataclasses import dataclass
from typing import List

import yahp as hp

from composer.models.model_hparams import Initializer, ModelHparams


@dataclass
class DeepLabv3Hparams(ModelHparams):
    ignore_index: int = hp.optional("Class index to ignore for loss calculations", default=0)
    is_pretrained: str = hp.optional("Which pretrained backbone to use ['old', 'new']", default='')
    penult_kernel: int = hp.optional("Kernel size of penultimate convolution", default=1)
    sync_bn: bool = hp.optional("Whether or not to sync the batch statistics across devices", default=False)
    dropout: float = hp.optional("Whether or not to use dropout on penultimate activations", default=0.0)

    def initialize_object(self):
        from composer.models.deeplabv3.deeplabv3 import DeepLabv3
        return DeepLabv3(self)
