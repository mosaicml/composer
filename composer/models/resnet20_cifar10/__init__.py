# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.resnet20_cifar10.model import CIFAR10_ResNet20 as CIFAR10_ResNet20
from composer.models.resnet20_cifar10.resnet20_cifar10_hparams import CIFARResNet20Hparams as CIFARResNet20Hparams

_task = 'Image Classification'
_dataset = 'CIFAR10'
_name = 'ResNet20'
_quality = 'tbd'
_metric = 'Top-1 Accuracy'
_ttt = 'tbd'
_hparams = 'resnet20_cifar10.yaml'
