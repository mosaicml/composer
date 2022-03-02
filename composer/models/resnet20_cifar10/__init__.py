# Copyright 2021 MosaicML. All Rights Reserved.

"""A ResNet20 model for CIFAR10.

See the :doc:`Model Card </model_cards/cifar_resnet>` for more details.
"""
from composer.models.resnet20_cifar10.model import CIFAR10_ResNet20 as CIFAR10_ResNet20
from composer.models.resnet20_cifar10.resnet20_cifar10_hparams import CIFARResNet20Hparams as CIFARResNet20Hparams

__all__ = ["CIFAR10_ResNet20", "CIFARResNet20Hparams"]

_task = 'Image Classification'
_dataset = 'CIFAR10'
_name = 'ResNet20'
_quality = 'tbd'
_metric = 'Top-1 Accuracy'
_ttt = 'tbd'
_hparams = 'resnet20_cifar10.yaml'
