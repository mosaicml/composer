# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for
:func:`.composer_resnet_cifar`."""
from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.models.model_hparams import ModelHparams
from composer.models.resnet_cifar import composer_resnet_cifar

__all__ = ['ResNetCIFARHparams']


@dataclass
class ResNetCIFARHparams(ModelHparams):
    """:class:`~.hp.Hparams` interface for :func:`.composer_resnet_cifar`.

    Args:
        model_name (str): ``"resnet_9"``, ``"resnet_20"``, or ``"resnet_56"``.
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization. Default: ``None``.
    """
    model_name: Optional[str] = hp.optional('"cifar_resnet_9", "cifar_resnet_20" or "cifar_resnet_56"', default=None)
    num_classes: int = hp.optional('The number of classes.  Needed for classification tasks', default=10)

    def validate(self):
        if self.model_name is None:
            raise ValueError('model name must be one of "cifar_resnet_9", "cifar_resnet_20" or "cifar_resnet_56".')

    def initialize_object(self):
        if self.model_name is None:
            raise ValueError('model name must be one of "cifar_resnet_9", "cifar_resnet_20" or "cifar_resnet_56".')
        return composer_resnet_cifar(model_name=self.model_name,
                                     num_classes=self.num_classes,
                                     initializers=self.initializers)
