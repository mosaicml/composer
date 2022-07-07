# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.composer_deeplabv3`."""

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ['DeepLabV3Hparams']


@dataclass
class DeepLabV3Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for
        :func:`.composer_deeplabv3`.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either
            [``'resnet50'``, ``'resnet101'``]. Default: ``'resnet101'``.
        is_backbone_pretrained (bool, optional): If ``True``, use pretrained weights for the backbone.
            Default: ``True``.
        backbone_url (str, optional): Url used to download model weights. If empty, the PyTorch url will be used.
            Default: ``''``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers.
            Default: ``True``.
        initializers (List[Initializer], optional): Initializers for the model. ``[]`` for no initialization.
            Default: ``[]``.
    """

    backbone_arch: str = hp.optional("The backbone architecture to use. Must be either ['resnet50', resnet101'].",
                                     default='resnet101')
    is_backbone_pretrained: bool = hp.optional('If true, use pre-trained weights for backbone.', default=True)
    backbone_url: str = hp.optional(
        "Url to download model weights from. If blank (default), will download from PyTorch's url.", default='')
    use_plus: bool = hp.optional('If true (default), use DeepLabv3+ head instead of DeepLabv3.', default=True)
    sync_bn: bool = hp.optional('If true, use SyncBatchNorm to sync batch norm statistics across GPUs.', default=True)
    ignore_index: int = hp.optional('Class label to ignore when calculating the loss and other metrics.', default=-1)
    cross_entropy_weight: float = hp.optional('Weight to scale the cross entropy loss.', default=1.0)
    dice_weight: float = hp.optional('Weight to scale the dice loss.', default=0.0)

    def validate(self):
        if self.num_classes is None:
            raise ValueError('num_classes must be specified')

        if self.backbone_arch not in ['resnet50', 'resnet101']:
            raise ValueError(f"backbone_arch must be one of ['resnet50', 'resnet101']: not {self.backbone_arch}")

        if self.cross_entropy_weight < 0:
            raise ValueError(f'cross_entropy_weight value {self.cross_entropy_weight} must be positive or zero.')

        if self.dice_weight < 0:
            raise ValueError(f'dice_weight value {self.dice_weight} must be positive or zero.')

        if self.cross_entropy_weight == 0 and self.dice_weight == 0:
            raise ValueError('Both cross_entropy_weight and dice_weight cannot be zero.')

    def initialize_object(self):
        from composer.models.deeplabv3.model import composer_deeplabv3

        if self.num_classes is None:
            raise ValueError('num_classes must be specified')

        return composer_deeplabv3(num_classes=self.num_classes,
                                  backbone_arch=self.backbone_arch,
                                  is_backbone_pretrained=self.is_backbone_pretrained,
                                  backbone_url=self.backbone_url,
                                  use_plus=self.use_plus,
                                  sync_bn=self.sync_bn,
                                  ignore_index=self.ignore_index,
                                  cross_entropy_weight=self.cross_entropy_weight,
                                  dice_weight=self.dice_weight,
                                  initializers=self.initializers)
