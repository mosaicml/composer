# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

from composer.models.base import MosaicClassifier
from composer.models.model_hparams import Initializer
from composer.models.resnets import ImageNet_ResNet


class ResNet101(MosaicClassifier):
    """A ResNet-101 model extending :class:`MosaicClassifier`.

    See this `paper <https://arxiv.org/abs/1512.03385>`_ for details
    on the residual network architecture.

    Args:
        num_classes (int): The number of classes for the model.
        initializers (List[Initializer], optional): Initializers
            for the model. ``None`` for no initialization.
            (default: ``None``)
    """

    def __init__(
        self,
        num_classes: int,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:
        if initializers is None:
            initializers = []

        model = ImageNet_ResNet.get_model_from_name(
            "imagenet_resnet_101",
            initializers,
            num_classes,
        )
        super().__init__(module=model)
