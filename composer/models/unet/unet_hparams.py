# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ["UnetHparams"]


@dataclass
class UnetHparams(ModelHparams):
    """yahp Hparams interface for Unet segmentation models.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: 3.
    """

    def initialize_object(self):
        from composer.models.unet.unet import UNet
        return UNet(**asdict(self))
