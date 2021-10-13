# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.base import MosaicClassifier
from composer.models.efficientnetb0.efficientnetb0_hparams import EfficientNetB0Hparams
from composer.models.efficientnets import EfficientNet


class EfficientNetB0(MosaicClassifier):

    def __init__(self, hparams: EfficientNetB0Hparams) -> None:
        model = EfficientNet.get_model_from_name(
            "efficientnet-b0",
            hparams.num_classes,
            hparams.drop_connect_rate,
        )
        super().__init__(module=model)
