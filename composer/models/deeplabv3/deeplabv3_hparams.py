from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams


@dataclass
class DeepLabv3Hparams(ModelHparams):
    num_classes: int = hp.required("Number of classes for classification")
    ignore_index: int = hp.optional("Class index to ignore for loss calculations", default=0)
    is_pretrained: bool = hp.optional("Whether or not to use a pretrained ImageNet backbone", default=False)

    def initialize_object(self):
        from composer.models.deeplabv3.deeplabv3 import DeepLabv3
        return DeepLabv3(self)
