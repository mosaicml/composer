# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class ResNet18Hparams(ModelHparams):

    def initialize_object(self):
        from composer.models import ResNet18
        return ResNet18(**asdict(self))
