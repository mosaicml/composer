# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.composer_resnet`."""

from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.models.model_hparams import ModelHparams
from composer.models.resnet.model import composer_resnet, valid_model_names

__all__ = ['ResNetHparams']


@dataclass
class ResNetHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.composer_resnet`.

    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        weights (str, optional): If provided, pretrained weights can be specified, such as with ``IMAGENET1K_V2``. Default: ``None``.
        pretrained (bool, optional): If True, use ImageNet pretrained weights. Default: ``False``. This parameter is deprecated and
            will soon be removed.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.
    """

    model_name: str = hp.optional(
        f"ResNet architecture to instantiate, must be one of {valid_model_names}. (default: '')", default='')
    weights: Optional[str] = hp.optional(
        'If provided, pretrained weights can be specified, such as with ``IMAGENET1K_V2``. (default: ``None``)',
        default=None)
    pretrained: bool = hp.optional(
        'If True, use ImageNet pretrained weights. Default: ``False``. This parameter is deprecated and will soon be removed.',
        default=False)
    groups: int = hp.optional(
        'Number of filter groups for the 3x3 convolution layer in bottleneck block. (default: ``1``)', default=1)
    width_per_group: int = hp.optional(
        'Initial width for each convolution group. Width doubles after each stage. (default: ``64``)', default=64)
    loss_name: str = hp.optional(
        "Name of loss function. E.g. 'soft_cross_entropy' or 'binary_cross_entropy_with_logits'. (default: ``soft_cross_entropy``)",
        default='soft_cross_entropy')

    def validate(self):
        if self.num_classes is None:
            raise ValueError('num_classes must be specified')

    def initialize_object(self):
        if self.num_classes is None:
            raise ValueError('num_classes must be specified')
        return composer_resnet(model_name=self.model_name,
                               num_classes=self.num_classes,
                               weights=self.weights,
                               pretrained=self.pretrained,
                               groups=self.groups,
                               width_per_group=self.width_per_group,
                               initializers=self.initializers,
                               loss_name=self.loss_name)
