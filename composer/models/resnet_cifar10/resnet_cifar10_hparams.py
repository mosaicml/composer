# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.CIFAR10ResNet`."""
from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ["CIFAR10ResNetHparams"]


@dataclass
class CIFAR10ResNetHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.CIFAR10ResNet`.

    Args:
        model_name (str): ``"cifar_resnet_9"``, ``"cifar_resnet_20"``, or ``"cifar_resnet_56"``.
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization. Default: ``None``.
    """
    model_name: str = hp.optional('"cifar_resnet_9", "cifar_resnet_20" or "cifar_resnet_56"', default=None)
    num_classes: int = hp.optional("The number of classes.  Needed for classification tasks", default=10)

    def validate(self):
        if self.model_name is None:
            raise ValueError('model name must be one of "cifar_resnet_9", "cifar_resnet_20" or "cifar_resnet_56".')

    def initialize_object(self):
        from composer.models import CIFAR10ResNet
        return CIFAR10ResNet(model_name=self.model_name, num_classes=self.num_classes, initializers=self.initializers)
