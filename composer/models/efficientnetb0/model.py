# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.base import ComposerClassifier
from composer.models.efficientnets import EfficientNet


class EfficientNetB0(ComposerClassifier):
    """A :class:`ComposerClassifier` wrapper around the EfficientNet-b0 architecture.

    From the paper EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks `<https://arxiv.org/abs/1905.11946>`_.

    Example:

    .. testcode::

        from composer.models import EfficientNetB0

        model = EfficientNetB0()  # creates EfficientNet-b0 for image classification

    Args:
        num_classes (int): The number of classes. Needed for classification tasks. Default = 1000.
        drop_connect_rate (float): Probability of dropping a sample within a block before identity connection. Default = 0.2
    """

    def __init__(self, num_classes: int = 1000, drop_connect_rate: float = 0.2) -> None:
        model = EfficientNet.get_model_from_name(
            "efficientnet-b0",
            num_classes,
            drop_connect_rate,
        )
        super().__init__(module=model)
