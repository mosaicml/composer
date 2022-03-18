# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.CIFAR10_ResNet9`."""

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ["CIFARResNet9Hparams"]


@dataclass
class CIFARResNet9Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.CIFAR10_ResNet9`.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
    """
    num_classes: int = hp.optional("The number of classes.  Needed for classification tasks", default=10)

    def initialize_object(self):
        from composer.models import CIFAR10_ResNet9
        return CIFAR10_ResNet9(num_classes=self.num_classes)
