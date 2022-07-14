# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A :class:`.ComposerClassifier` wrapper around the EfficientNet-b0 architecture."""
from composer.models.efficientnetb0.efficientnets import EfficientNet
from composer.models.tasks import ComposerClassifier

__all__ = ['composer_efficientnetb0']


def composer_efficientnetb0(num_classes: int = 1000, drop_connect_rate: float = 0.2) -> ComposerClassifier:
    """Helper function to create a :class:`.ComposerClassifier` with an EfficientNet-b0 architecture.

    See `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_
        (Tan et al, 2019) for more details.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        drop_connect_rate (float, optional): Probability of dropping a sample within a block before identity
            connection. Default: ``0.2``.

    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with a EfficientNet-B0 model.


    Example:

    .. testcode::

        from composer.models import composer_efficientnetb0

        model = composer_efficientnetb0()  # creates EfficientNet-b0 for image classification
    """
    model = EfficientNet.get_model_from_name(model_name='efficientnet-b0',
                                             num_classes=num_classes,
                                             drop_connect_rate=drop_connect_rate)

    composer_model = ComposerClassifier(module=model)
    return composer_model
