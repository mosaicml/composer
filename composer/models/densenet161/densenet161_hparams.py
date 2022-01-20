# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
import yahp as hp
from composer.models.model_hparams import ModelHparams
from composer.models.densenet161 import DenseNet161


@dataclass
class DenseNet161Hparams(ModelHparams):
    
    num_channels: int = hp.optional("number of  image channels", default=3)
    num_classes: int = hp.optional("The number of classes.  Needed for classification tasks", default=1000)
    
    dropout: float = hp.optional("dropout rate", default=0.0)
    pretrained: bool = hp.optional("imagenet pretrained", default=False)

    def validate(self):
        if self.num_channels is None:
            raise ValueError("channels must be specified")
        if self.num_classes is None:
            raise ValueError("num_classes must be specified")

    def initialize_object(self):
        return DenseNet161(num_classes=self.num_classes,
                            num_channels=self.num_channels,
                            dropout=self.dropout,
                            pretrained=self.pretrained)