# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
import yahp as hp


@dataclass
class ViTSmallPatch16Hparams(hp.Hparams):
    
    image_size: int = hp.required("input image size. If you have rectangular images, make sure your image size is the maximum of the width and height")
    channels: int = hp.required("number of  image channels")
    num_classes: int = hp.required("The number of classes.  Needed for classification tasks")
    
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

