# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported datasets."""

from composer.datasets.ade20k import ADE20k, StreamingADE20k
from composer.datasets.brats import PytTrain, PytVal
from composer.datasets.c4 import C4Dataset
from composer.datasets.coco import COCODetection, StreamingCOCO
from composer.datasets.imagenet import StreamingImageNet1k
from composer.datasets.synthetic import (SyntheticBatchPairDataset, SyntheticDataLabelType, SyntheticDataType,
                                         SyntheticPILDataset)

__all__ = [
    "ADE20k", "StreamingADE20k", "PytTrain", "PytVal", "C4Dataset", "COCODetection", "StreamingCOCO",
    "StreamingImageNet1k", "SyntheticBatchPairDataset", "SyntheticDataLabelType", "SyntheticDataType",
    "SyntheticPILDataset"
]
