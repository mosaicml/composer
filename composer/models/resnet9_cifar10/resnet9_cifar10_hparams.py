# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ["CIFARResNet9Hparams"]


@dataclass
class CIFARResNet9Hparams(ModelHparams):

    def initialize_object(self):
        from composer.models import CIFAR10_ResNet9
        return CIFAR10_ResNet9(**asdict(self))
