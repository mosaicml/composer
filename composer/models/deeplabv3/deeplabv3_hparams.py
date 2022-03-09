# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ["DeepLabV3Hparams"]


@dataclass
class DeepLabV3Hparams(ModelHparams):
    """yahp interface for :class:`~.ComposerDeepLabV3`. This class specifies arguments for a DeepLabV3 model and can
    instantiate a DeepLabV3 model.

    Args:
        num_classes (int): The number of classes in the segmentation task.
        backbone_arch (str, optional): The backbone architecture to use, either 'resnet50', 'resnet101'. Default: ``'resnet101'``.
        is_backbone_pretrained (bool, optional): if true (default), use pre-trained weights for backbone. Default: True.
        sync_bn (bool, optional): Use SyncBatchNorm to sync batch norm statistics across GPUs. Default: True.
        initializers (List[Initializer], optional): Initializers for the model. [] for no initialization. Default: [].
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
