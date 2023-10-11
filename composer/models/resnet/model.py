# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A :class:`.ComposerClassifier` wrapper around the torchvision implementations of the ResNet model family."""

import logging
import warnings
from typing import List, Optional

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import resnet

from composer.loss import loss_registry
from composer.metrics import CrossEntropy
from composer.models.initializers import Initializer
from composer.models.tasks import ComposerClassifier

__all__ = ['composer_resnet']

log = logging.getLogger(__name__)

valid_model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def composer_resnet(model_name: str,
                    num_classes: int = 1000,
                    weights: Optional[str] = None,
                    groups: int = 1,
                    width_per_group: int = 64,
                    initializers: Optional[List[Initializer]] = None,
                    loss_name: str = 'soft_cross_entropy') -> ComposerClassifier:
    """Helper function to create a :class:`.ComposerClassifier` with a torchvision ResNet model.

    From `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_ (He et al, 2015).

    Args:
        model_name (str): Name of the ResNet model instance. Either [``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``,
            ``"resnet152"``].
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        weights (str, optional): If provided, pretrained weights can be specified, such as with ``IMAGENET1K_V2``. Default: ``None``.
        groups (int, optional): Number of filter groups for the 3x3 convolution layer in bottleneck blocks. Default: ``1``.
        width_per_group (int, optional): Initial width for each convolution group. Width doubles after each stage.
            Default: ``64``.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization.
            Default: ``None``.
        loss_name (str, optional): Loss function to use. E.g. 'soft_cross_entropy' or
            'binary_cross_entropy_with_logits'. Loss function must be in
            :mod:`~composer.loss.loss`. Default: ``'soft_cross_entropy'``".
    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with a torchvision ResNet model.

    Example:

    .. testcode::

        from composer.models import composer_resnet

        model = composer_resnet(model_name='resnet18')  # creates a torchvision resnet18 for image classification
    """
    warnings.warn(DeprecationWarning('composer_resnet is deprecated and will be removed in v0.18'))

    valid_model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if model_name not in valid_model_names:
        raise ValueError(f'model_name must be one of {valid_model_names} instead of {model_name}.')

    if loss_name not in loss_registry.keys():
        raise ValueError(f'Unrecognized loss function: {loss_name}. Please ensure the '
                         'specified loss function is present in composer.loss.loss.py')

    if loss_name == 'binary_cross_entropy_with_logits' and (initializers is None or
                                                            Initializer.LINEAR_LOG_CONSTANT_BIAS not in initializers):
        log.warning('UserWarning: Using `binary_cross_entropy_loss_with_logits` '
                    'without using `initializers.linear_log_constant_bias` can degrade '
                    'performance. '
                    'Please ensure you are using `initializers. '
                    'linear_log_constant_bias`.')

    if initializers is None:
        initializers = []

    # Instantiate model
    model_fn = getattr(resnet, model_name)
    model = model_fn(weights=weights, num_classes=num_classes, groups=groups, width_per_group=width_per_group)

    # Grab loss function from loss registry
    loss_fn = loss_registry[loss_name]

    # Create metrics for train and validation
    train_metrics = MulticlassAccuracy(num_classes=num_classes, average='micro')
    val_metrics = MetricCollection([CrossEntropy(), MulticlassAccuracy(num_classes=num_classes, average='micro')])

    # Apply Initializers to model
    for initializer in initializers:
        initializer = Initializer(initializer)
        model.apply(initializer.get_initializer())

    composer_model = ComposerClassifier(model, train_metrics=train_metrics, val_metrics=val_metrics, loss_fn=loss_fn)
    return composer_model
