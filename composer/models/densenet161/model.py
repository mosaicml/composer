# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

from composer.models.base import MosaicClassifier
from composer.models.model_hparams import Initializer
import torchvision
from torch import nn


class DenseNet161(MosaicClassifier):
    """Torchvision Densenet161 https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py"""

    def __init__(self,
                num_classes: int,
                num_channels: int,
                dropout: float = 0.0,
                pretrained: bool = False) -> None:


        model = torchvision.models.densenet161(pretrained=pretrained, drop_rate=dropout)
        if num_channels != 3:
            model.features[0] = nn.Conv2d(
                num_channels,
                out_channels=96,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        if num_classes != 1000:
            model.classifier = nn.Linear(
                in_features=2208, out_features=num_classes, bias=True
            )
        super().__init__(module=model)
