from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams


@dataclass
class DeepLabV3Hparams(ModelHparams):
    """This class specifies arguments for a DeepLabV3 model and can instantiate a DeepLabV3 model.
    """
    backbone_arch: str = hp.optional("The backbone architecture to use. Must be either ['resnet50', resnet101']",
                                     default='resnet101')
    is_backbone_pretrained: bool = hp.optional("Whether or not to use a pretrained backbone", default=True)
    sync_bn: bool = hp.optional("Whether or not to sync the batch statistics across devices", default=False)

    def validate(self):
        if self.num_classes is None:
            raise ValueError("num_classes must be specified")

        if self.backbone_arch not in ['resnet50', 'resnet101']:
            raise ValueError(f"backbone_arch must be one of ['resnet50', 'resnet101']: not {self.backbone_arch}")

    def initialize_object(self):
        from composer.models.deeplabv3.deeplabv3 import MosaicDeepLabV3
        return MosaicDeepLabV3(
            num_classes=self.num_classes,  # type: ignore
            backbone_arch=self.backbone_arch,
            is_backbone_pretrained=self.is_backbone_pretrained,
            sync_bn=self.sync_bn)
