# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ViT Small Patch 16 for image classification."""

from composer.models.vit_small_patch16.model import vit_small_patch16 as vit_small_patch16

__all__ = ['vit_small_patch16']

_task = 'Image Classification'
_dataset = 'ImageNet'
_name = 'ViT-Small-Patch16'
_quality = '74.52'
_metric = 'Top-1 Accuracy'
_ttt = '1d 59m'
_hparams = 'vit_small_patch16.yaml'
