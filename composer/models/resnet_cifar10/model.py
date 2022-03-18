# Copyright 2021 MosaicML. All Rights Reserved.

"""ResNet models for CIFAR10 extending :class:`.ComposerClassifier`."""

from typing import List, Optional

from composer.models.base import ComposerClassifier
from composer.models.model_hparams import Initializer
from composer.models.resnet_cifar10.resnets import CIFARResNet, ResNet9

__all__ = ["CIFAR10ResNet"]


class CIFAR10ResNet(ComposerClassifier):
    """ResNet models for CIFAR10 extending :class:`.ComposerClassifier`.

    From `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_ (He et al, 2015).

    Args:
        model_name (str): ``"cifar_resnet_9"``, ``"cifar_resnet_20"``, or ``"cifar_resnet_56"``.
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization. Default: ``None``.

    Example:

    .. testcode::

        from composer.models import CIFAR10ResNet

        model = CIFAR10ResNet(model_name="cifar_resnet_56")  # creates a resnet56 for cifar image classification
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:
        if initializers is None:
            initializers = []

        if model_name == "cifar_resnet_9":
            model = ResNet9(num_classes)  # current initializers don't work with this architecture.
        else:
            model = CIFARResNet.get_model_from_name(
                model_name,
                initializers,
                num_classes,
            )
        super().__init__(module=model)
