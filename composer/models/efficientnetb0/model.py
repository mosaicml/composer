# Copyright 2021 MosaicML. All Rights Reserved.

"""A :class:`.ComposerClassifier` wrapper around the EfficientNet-b0 architecture."""
from composer.models.base import ComposerClassifier
from composer.models.efficientnetb0.efficientnets import EfficientNet

__all__ = ["EfficientNetB0"]


class EfficientNetB0(ComposerClassifier):
    """A :class:`.ComposerClassifier` wrapper around the EfficientNet-b0 architecture. From `Rethinking Model Scaling
    for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_ (Tan et al, 2019).

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        drop_connect_rate (float, optional): Probability of dropping a sample within a block before identity connection. Default: ``0.2``.

    Example:

    .. testcode::

        from composer.models import EfficientNetB0

        model = EfficientNetB0()  # creates EfficientNet-b0 for image classification
    """

    def __init__(self, num_classes: int = 1000, drop_connect_rate: float = 0.2) -> None:
        model = EfficientNet.get_model_from_name(
            "efficientnet-b0",
            num_classes,
            drop_connect_rate,
        )
        super().__init__(module=model)
