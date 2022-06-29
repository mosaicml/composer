# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""DeepLabV3 for image segmentation."""
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabV3Hparams as DeepLabV3Hparams
from composer.models.deeplabv3.model import composer_deeplabv3 as composer_deeplabv3

__all__ = ['composer_deeplabv3', 'DeepLabV3Hparams']
