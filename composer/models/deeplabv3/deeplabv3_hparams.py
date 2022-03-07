# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams


@dataclass
class DeepLabV3Hparams(ModelHparams):
    """This class specifies arguments for a DeepLabV3 model and can instantiate a DeepLabV3 model.

    Args:
        backbone_arch (str): the backbone architecture to use, either ['resnet50', 'resnet101'].
            Default is 'resnet101'.
        is_backbone_pretrained (bool): if true (default), use pre-trained weights for backbone.
        backbone_url (str): url to download model weights from. If blank (default), will download from PyTorch's url.
        use_plus (bool): if true (default), use DeepLabv3+ head instead of DeepLabv3.
        sync_bn (bool): if true (default), use SyncBatchNorm to sync batch norm statistics across GPUs.
    """
    backbone_arch: str = hp.optional("The backbone architecture to use. Must be either ['resnet50', resnet101'].",
                                     default='resnet101')
    is_backbone_pretrained: bool = hp.optional("If true, use pre-trained weights for backbone.", default=True)
    backbone_url: str = hp.optional(
        "Url to download model weights from. If blank (default), will download from PyTorch's url.", default='')
    use_plus: bool = hp.optional("If true (default), use DeepLabv3+ head instead of DeepLabv3.", default=True)
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
                                 backbone_url=self.backbone_url,
                                 use_plust=self.use_plus,
                                 sync_bn=self.sync_bn,
                                 initializers=self.initializers)
