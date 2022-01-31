# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp

from composer.models import MosaicResNet
from composer.models.model_hparams import ModelHparams


@dataclass
class ResNetHparams(ModelHparams):
    model_name: str = hp.optional(
        f"ResNet architecture to instantiate, must be one of"
        f"{MosaicResNet.valid_model_names}. Default is ''.",
        default='')
    num_classes: int = hp.optional("Number of classes for the classification taks. Default is None.", default=None)
    pretrained: bool = hp.optional("If true, use ImageNet pretrained weights. Default is False.", default=False)
    groups: int = hp.optional(
        "Number of filter groups for the 3x3 convolution layer in the bottleneck block."
        "Default is 1.", default=1)
    width_per_group: int = hp.optional(
        "Initial number of filters for each convolution group. The number of filters"
        "doubles at each stage in the network. Default is 64",
        default=64)

    def validate(self):
        if self.model_name not in MosaicResNet.valid_model_names:
            raise ValueError(f"model_name must be one of {MosaicResNet.valid_model_names}, but got {self.model_name}")

        if self.num_classes is None:
            raise ValueError("num_classes must be specified")

    def initialize_object(self):
        return MosaicResNet(model_name=self.model_name,
                            num_classes=self.num_classes,
                            pretrained=self.pretrained,
                            groups=self.groups,
                            width_per_group=self.width_per_group)
