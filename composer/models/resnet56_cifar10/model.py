# Copyright 2021 MosaicML. All Rights Reserved.

"""A ResNet-56 model extending :class:`.ComposerClassifier`."""

from typing import List, Optional

from composer.models.base import ComposerClassifier
from composer.models.model_hparams import Initializer
from composer.models.resnets import CIFAR_ResNet

__all__ = ["CIFAR10_ResNet56"]


class CIFAR10_ResNet56(ComposerClassifier):
    """A ResNet-56 model extending :class:`.ComposerClassifier`.

    From `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_ (He et al, 2015).

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization. Default: ``None``.

    Example:

    .. testcode::

        from composer.models import CIFAR10_ResNet56

        model = CIFAR10_ResNet56()  # creates a resnet56 for cifar image classification
    """

    def __init__(
        self,
        num_classes: int = 10,
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
