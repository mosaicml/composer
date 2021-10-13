# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

from composer.models.base import MosaicClassifier
from composer.models.model_hparams import Initializer
from composer.models.resnets import CIFAR_ResNet


class CIFAR10_ResNet56(MosaicClassifier):

    def __init__(
        self,
        num_classes: int,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:
        if initializers is None:
            initializers = []

        model = CIFAR_ResNet.get_model_from_name(
            "cifar_resnet_56",
            initializers,
            num_classes,
        )
        super().__init__(module=model)
