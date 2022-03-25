# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerResNet`."""

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams
from composer.models.resnet.model import ComposerResNet

__all__ = ["ResNetHparams"]


@dataclass
class ResNetHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerResNet`.

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
    """

    model_name: str = hp.optional(
        f"ResNet architecture to instantiate, must be one of {ComposerResNet.valid_model_names}. (default: '')",
        default='')
    num_classes: int = hp.optional("Number of classes for the classification taks. (default: ``None``)", default=None)
    pretrained: bool = hp.optional("If true, use ImageNet pretrained weights. (default: ``False``)", default=False)
    groups: int = hp.optional(
        "Number of filter groups for the 3x3 convolution layer in bottleneck block. (default: ``1``)", default=1)
    width_per_group: int = hp.optional(
        "Initial width for each convolution group. Width doubles after each stage. (default: ``64``)", default=64)

    def validate(self):
        if self.model_name not in ComposerResNet.valid_model_names:
            raise ValueError(f"model_name must be one of {ComposerResNet.valid_model_names}, but got {self.model_name}")

        if self.num_classes is None:
            raise ValueError("num_classes must be specified")

    def initialize_object(self):
        return ComposerResNet(model_name=self.model_name,
                              num_classes=self.num_classes,
                              pretrained=self.pretrained,
                              groups=self.groups,
                              width_per_group=self.width_per_group,
                              initializers=self.initializers)
