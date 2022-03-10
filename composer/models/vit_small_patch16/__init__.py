# Copyright 2021 MosaicML. All Rights Reserved.

"""ViT Small Patch 16 for image classification."""

from composer.models.vit_small_patch16.hparams import ViTSmallPatch16Hparams as ViTSmallPatch16Hparams
from composer.models.vit_small_patch16.model import ViTSmallPatch16 as ViTSmallPatch16

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'ViT-Small-Patch16'
_quality = '74.52'
_metric = 'Top-1 Accuracy'
_ttt = '1d 59m'
_hparams = 'vit_small_patch16.yaml'
