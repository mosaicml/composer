# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.unet.unet import UNet as UNet
from composer.models.unet.unet_hparams import UnetHparams as UnetHparams

_task = 'Image Segmentation'
_dataset = 'BRATS'
_name = 'UNet'
_quality = '69.1'
_metric = 'Dice'
_ttt = '21m'
_hparams = 'unet.yaml'
