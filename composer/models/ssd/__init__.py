# Copyright 2021 MosaicML. All Rights Reserved.

"""SSD300 for object detection on MSCOCO."""

from composer.models.ssd.ssd import SSD as SSD
from composer.models.ssd.ssd_hparams import SSDHparams as SSDHparams

_task = 'Object Detection'
_dataset = 'COCO'
_name = 'SSD'
_hparams = 'ssd.yaml'
