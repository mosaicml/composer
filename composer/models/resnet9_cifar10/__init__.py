# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.resnet9_cifar10.model import CIFAR10_ResNet9 as CIFAR10_ResNet9
from composer.models.resnet9_cifar10.resnet9_cifar10_hparams import CIFARResNet9Hparams as CIFARResNet9Hparams

_task = 'Image Classification'
_dataset = 'CIFAR10'
_name = 'ResNet9'
_quality = '92.9'
_metric = 'Top-1 Accuracy'
_ttt = '5m'
_hparams = 'resnet9_cifar10.yaml'
