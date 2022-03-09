# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerDeepLabV3`."""

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ["DeepLabV3Hparams"]


@dataclass
class DeepLabV3Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ComposerDeepLabV3`.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either [``'resnet50'``, ``'resnet101'``].
            Default: ``'resnet101'``.
        is_backbone_pretrained (bool, optional): If ``True``, use pretrained weights for the backbone. Default: ``True``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers. Default: ``True``.
        initializers (List[Initializer], optional): Initializers for the model. ``[]`` for no initialization. Default: ``[]``.
    """

    backbone_arch: str = hp.optional("The backbone architecture to use. Must be either ['resnet50', resnet101'].",
                                     default='resnet101')
    is_backbone_pretrained: bool = hp.optional("If true, use pre-trained weights for backbone.", default=True)
    sync_bn: bool = hp.optional("If true, use SyncBatchNorm to sync batch norm statistics across GPUs.", default=True)

    def validate(self):
        if self.num_classes is None:
            raise ValueError("num_classes must be specified")

        if self.backbone_arch not in ['resnet50', 'resnet101']:
            raise ValueError(f"backbone_arch must be one of ['resnet50', 'resnet101']: not {self.backbone_arch}")

    def initialize_object(self):
        from composer.models.deeplabv3.deeplabv3 import ComposerDeepLabV3

        if self.num_classes is None:
            raise ValueError("num_classes must be specified")

        return ComposerDeepLabV3(num_classes=self.num_classes,
                                 backbone_arch=self.backbone_arch,
                                 is_backbone_pretrained=self.is_backbone_pretrained,
                                 sync_bn=self.sync_bn,
                                 initializers=self.initializers)
