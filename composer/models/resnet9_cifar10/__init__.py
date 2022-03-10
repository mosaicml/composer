# Copyright 2021 MosaicML. All Rights Reserved.

"""A ResNet9 model for CIFAR10.

See the :doc:`Model Card </model_cards/cifar_resnet>` for more details.
"""
from composer.models.resnet9_cifar10.model import CIFAR10_ResNet9 as CIFAR10_ResNet9
from composer.models.resnet9_cifar10.resnet9_cifar10_hparams import CIFARResNet9Hparams as CIFARResNet9Hparams

__all__ = ["CIFAR10_ResNet9", "CIFARResNet9Hparams"]

_task = 'Image Classification'
_dataset = 'CIFAR10'
_name = 'ResNet9'
_quality = '92.9'
_metric = 'Top-1 Accuracy'
_ttt = '5m'
_hparams = 'resnet9_cifar10.yaml'
