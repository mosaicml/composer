# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.resnet18.model import ResNet18 as ResNet18
from composer.models.resnet18.resnet18_hparams import ResNet18Hparams as ResNet18Hparams

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'ResNet18'
_quality = ''
_metric = 'Top-1 Accuracy'
_ttt = '?'
_hparams = 'resnet18.yaml'
