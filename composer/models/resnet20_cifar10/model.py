# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

from composer.models.base import ComposerClassifier
from composer.models.model_hparams import Initializer
from composer.models.resnets import CIFAR_ResNet


class CIFAR10_ResNet20(ComposerClassifier):
    """A ResNet-20 model extending :class:`ComposerClassifier`.

    See this `paper <https://arxiv.org/abs/1512.03385>`_ for details
    on the residual network architecture.

    Args:
        num_classes (int): The number of classes for the model. Default = 10.
        initializers (List[Initializer], optional): Initializers
            for the model. ``None`` for no initialization.
            (default: ``None``)
    """

    def __init__(
        self,
        num_classes: int = 10,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:
        if initializers is None:
            initializers = []

        model = CIFAR_ResNet.get_model_from_name(
            "cifar_resnet_20",
            initializers,
            num_classes,
        )
        super().__init__(module=model)
