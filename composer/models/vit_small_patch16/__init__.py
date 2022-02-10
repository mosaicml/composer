# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.vit_small_patch16.hparams import ViTSmallPatch16Hparams
from composer.models.vit_small_patch16.model import ViTSmallPatch16

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'ViT-Small-Patch16'
_quality = '74.4'  # target
_metric = 'Top-1 Accuracy'
_ttt = ''
_hparams = ''
