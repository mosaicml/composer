# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

from torchvision.models import resnet

from composer.models.base import MosaicClassifier
from composer.models.model_hparams import Initializer


class MosaicResNet(MosaicClassifier):
    """A ResNet-50 model extending :class:`MosaicClassifier`.

    See this `paper <https://arxiv.org/abs/1512.03385>`_ for details
    on the residual network architecture.

    Args:
        num_classes (int): The number of classes for the model.
        initializers (List[Initializer], optional): Initializers
            for the model. ``None`` for no initialization.
            (default: ``None``)
    """

    valid_model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:

        if model_name not in resnet.__all__:
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
