# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ["ViTSmallPatch16Hparams"]


@dataclass
class ViTSmallPatch16Hparams(ModelHparams):
    """
    Args:
        num_classes (int): number of classes for the model.
        image_size (int): input image size. If you have rectangular images, make sure your image size is the maximum of the width and height.
        channels (int): number of  image channels.
        dropout (float): 0.0 - 1.0 dropout rate.
        embedding_dropout (float): 0.0 - 1.0 embedding dropout rate.
    """
    image_size: int = hp.optional(
        "input image size. If you have rectangular images, make sure your image size is the maximum of the width and height",
        default=244)
    channels: int = hp.optional("number of  image channels", default=3)
    num_classes: int = hp.optional("number of classes.  Needed for classification tasks", default=1000)

    dropout: float = hp.optional("dropout rate", default=0.0)
    embedding_dropout: float = hp.optional("embedding dropout rate", default=0.0)

    def validate(self):
        if self.image_size is None:
            raise ValueError("image_size must be specified")
        if self.channels is None:
            raise ValueError("channels must be specified")
        if self.num_classes is None:
            raise ValueError("num_classes must be specified")

    def initialize_object(self):
        from composer.models import ViTSmallPatch16
        return ViTSmallPatch16(num_classes=self.num_classes,
                               image_size=self.image_size,
                               channels=self.channels,
                               dropout=self.dropout,
                               embedding_dropout=self.embedding_dropout)
