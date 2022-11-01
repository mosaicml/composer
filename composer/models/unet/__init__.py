# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Unet architecture used in image segmentation. The example we are using is for BRATS medical brain tumor dataset.

See the :doc:`Model Card </model_cards/unet>` for more details.
"""

from composer.models.unet.unet import UNet as UNet

__all__ = ['UNet']

_task = 'Image Segmentation'
_dataset = 'BRATS'
_name = 'UNet'
_quality = '69.1'
_metric = 'Dice'
_ttt = '21m'
_hparams = 'unet.yaml'
