# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class SSDHparams(ModelHparams):
    input_size: int = hp.optional(
        doc="input size",
        default=300,
    )
    
    def initialize_object(self):
        from composer.models.ssd.ssd import SSD
        return SSD(input_size=self.input_size)#self)#drop_connect_rate=self.drop_connect_rate)
