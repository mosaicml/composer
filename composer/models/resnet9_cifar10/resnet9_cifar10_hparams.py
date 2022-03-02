# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ["CIFARResNet9Hparams"]


@dataclass
class CIFARResNet9Hparams(ModelHparams):
    """yahp Hparams interface for CIFAR10_ResNet models.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: 10.
        initializers (List[Initializer], optional): Initializers
            for the model. ``None`` for no initialization.
            (default: ``None``).
    """

    def initialize_object(self):
        from composer.models import CIFAR10_ResNet9
        return CIFAR10_ResNet9(**asdict(self))
