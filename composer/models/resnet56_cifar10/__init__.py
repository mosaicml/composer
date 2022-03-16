# Copyright 2021 MosaicML. All Rights Reserved.

"""A ResNet56 model for CIFAR10.

See the :doc:`Model Card </model_cards/cifar_resnet>` for more details.
"""

from composer.models.resnet56_cifar10.model import CIFAR10_ResNet56 as CIFAR10_ResNet56
from composer.models.resnet56_cifar10.resnet56_cifar10_hparams import CIFARResNetHparams as CIFARResNetHparams

__all__ = ["CIFAR10_ResNet56", "CIFARResNetHparams"]

_task = 'Image Classification'
_dataset = 'CIFAR10'
_name = 'ResNet56'
_quality = '93.1'
_metric = 'Top-1 Accuracy'
_ttt = 'tbd'
_hparams = 'resnet56_cifar10.yaml'
