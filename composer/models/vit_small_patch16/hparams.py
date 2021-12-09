# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class ViTSmallPatch16Hparams(ModelHparams):

    def initialize_object(self):
        from composer.models import ViTSmallPatch16
        return ViTSmallPatch16(**asdict(self))
