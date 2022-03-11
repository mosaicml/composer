# Copyright 2021 MosaicML. All Rights Reserved.

"""A :class:`.ComposerClassifier` wrapper around the torchvision implementations of the ResNet model family."""

from typing import List, Optional

from torchvision.models import resnet

from composer.models.base import ComposerClassifier
from composer.models.model_hparams import Initializer

__all__ = ["ComposerResNet"]


class ComposerResNet(ComposerClassifier):
    """A :class:`.ComposerClassifier` wrapper around the torchvision implementations of the ResNet model family.

    From `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_ (He et al, 2015).

    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        pretrained (bool, optional): If True, use ImageNet pretrained weights. Default: ``False``.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.

    Example:

    .. testcode::

        from composer.models import ComposerResNet

        model = ComposerResNet(model_name='resnet18')  # creates a torchvision resnet18 for image classification
    """

    valid_model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:

        if model_name not in self.valid_model_names:
            raise ValueError(f"model_name must be one of {self.valid_model_names} instead of {model_name}.")

        if initializers is None:
            initializers = []

        model_func = getattr(resnet, model_name)
        model = model_func(pretrained=pretrained,
                           num_classes=num_classes,
                           groups=groups,
                           width_per_group=width_per_group)

        for initializer in initializers:
            initializer = Initializer(initializer)
            model.apply(initializer.get_initializer())

        super().__init__(module=model)
