# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A :class:`.ComposerClassifier` wrapper around the torchvision implementations of the ResNet model family."""

import logging
import warnings
from typing import List, Optional

import torchvision
from packaging import version
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
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
                    pretrained: bool = False,
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
        pretrained (bool, optional): If True, use ImageNet pretrained weights. Default: ``False``. This parameter is deprecated and
            will soon be removed in favor of ``weights``.
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

    # Configure pretrained/weights based on torchvision version
    if pretrained and weights:
        raise ValueError(
            'composer_resnet expects only one of ``pretrained`` or ``weights`` to be specified, but both were specified.'
        )
    if pretrained:
        weights = 'IMAGENET1K_V2'
        warnings.warn(
            DeprecationWarning(
                'The ``pretrained`` argument for composer_resnet is deprecated and will be removed in the future. Please use ``weights`` instead.'
            ))

    # Instantiate model
    model_fn = getattr(resnet, model_name)
    model = None
    if version.parse(torchvision.__version__) < version.parse('0.13.0'):
        if weights:
            pretrained = True
            warnings.warn(
                f'The current torchvision version {torchvision.__version__} does not support the ``weights`` argument, so ``pretrained=True`` will be used instead. To enable ``weights``, please ugprade to the latest version of torchvision.'
            )
        model = model_fn(pretrained=pretrained, num_classes=num_classes, groups=groups, width_per_group=width_per_group)
    else:
        model = model_fn(weights=weights, num_classes=num_classes, groups=groups, width_per_group=width_per_group)

    # Grab loss function from loss registry
    loss_fn = loss_registry[loss_name]

    # Create metrics for train and validation
    train_metrics = Accuracy()
    val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

    # Apply Initializers to model
    for initializer in initializers:
        initializer = Initializer(initializer)
        model.apply(initializer.get_initializer())

    composer_model = ComposerClassifier(model, train_metrics=train_metrics, val_metrics=val_metrics, loss_fn=loss_fn)
    return composer_model
