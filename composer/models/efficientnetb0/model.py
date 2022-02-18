# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.base import ComposerClassifier
from composer.models.efficientnets import EfficientNet


class EfficientNetB0(ComposerClassifier):
    """An EfficientNet-b0 model extending :class:`ComposerClassifier`.

    Based off of this `paper <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): the number of classes in the task.
        drop_connect_rate (float): probability of dropping a sample within a block before identity connection.
    """

    def __init__(self, num_classes: int, drop_connect_rate: float = 0.2) -> None:
        model = EfficientNet.get_model_from_name(
            "efficientnet-b0",
            num_classes,
            drop_connect_rate,
        )
        super().__init__(module=model)
