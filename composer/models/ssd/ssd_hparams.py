# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class SSDHparams(ModelHparams):

    def initialize_object(self):
        from composer.models.ssd.ssd import SSD
        return SSD(self)
