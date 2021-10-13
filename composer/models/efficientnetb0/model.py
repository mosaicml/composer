# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.base import MosaicClassifier
from composer.models.efficientnetb0.efficientnetb0_hparams import EfficientNetB0Hparams
from composer.models.efficientnets import EfficientNet


class EfficientNetB0(MosaicClassifier):
    """An EfficientNet-b0 model extending :class:`MosaicClassifier`.

    Based off of this `paper <https://arxiv.org/abs/1905.11946>`_.

    Args:
        hparams (EfficientNetB0Hparams): The hyperparameters for the model.
    """

    def __init__(self, hparams: EfficientNetB0Hparams) -> None:
        model = EfficientNet.get_model_from_name(
            "efficientnet-b0",
            hparams.num_classes,
            hparams.drop_connect_rate,
        )
        super().__init__(module=model)
