# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass
import yahp as hp


from composer.models.model_hparams import ModelHparams


@dataclass
class ViTSmallPatch16Hparams(ModelHparams):
    
    dropout: float = hp.optional("dropout rate", default=0.0)
    embedding_dropout: float = hp.optional("embedding dropout rate", default=0.0)

    def validate(self):
        if self.num_classes is None:
            raise ValueError("num_classes must be specified")
        if self.image_size is None:
            raise ValueError("image_size must be specified")
        if self.channels is None:
            raise ValueError("channels must be specified")

    def initialize_object(self):
        from composer.models import ViTSmallPatch16
        return ViTSmallPatch16(**asdict(self))
