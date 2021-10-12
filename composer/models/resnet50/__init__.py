# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.resnet50.model import ResNet50 as ResNet50
from composer.models.resnet50.resnet50_hparams import ResNet50Hparams as ResNet50Hparams

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'ResNet50'
_quality = 76.5
_metric = 'Top-1 Accuracy'
_ttt = '?'
_hparams = 'hparams.yaml'
