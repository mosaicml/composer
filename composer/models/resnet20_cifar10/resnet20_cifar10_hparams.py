# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.CIFAR10_ResNet20`."""

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ["CIFARResNet20Hparams"]


@dataclass
class CIFARResNet20Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.CIFAR10_ResNet20`.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization. Default: ``None``.
    """

    def initialize_object(self):
        from composer.models import CIFAR10_ResNet20
        return CIFAR10_ResNet20(**asdict(self))
