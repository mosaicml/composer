# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class CIFARResNet20Hparams(ModelHparams):

    def initialize_object(self):
        from composer.models import CIFAR10_ResNet20
        return CIFAR10_ResNet20(**asdict(self))
