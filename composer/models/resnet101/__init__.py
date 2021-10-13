# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.resnet101.model import ResNet101 as ResNet101
from composer.models.resnet101.resnet101_hparams import ResNet101Hparams as ResNet101Hparams

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'ResNet101'
_quality = '78.10'
_metric = 'Top-1 Accuracy'
_ttt = '8h 15m'
_hparams = 'resnet101.yaml'
