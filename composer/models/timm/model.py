# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper around `timm.create_model() <https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model>`_
used to create :class:`.ComposerClassifier`."""

from typing import Optional

from composer.models.tasks import ComposerClassifier
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['composer_timm']


def composer_timm(model_name: str,
                  pretrained: bool = False,
                  num_classes: int = 1000,
                  drop_rate: float = 0.0,
                  drop_path_rate: Optional[float] = None,
                  drop_block_rate: Optional[float] = None,
                  global_pool: Optional[str] = None,
                  bn_momentum: Optional[float] = None,
                  bn_eps: Optional[float] = None) -> ComposerClassifier:
    """A wrapper around `timm.create_model() <https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-
    model>`_ used to create :class:`.ComposerClassifier`.

    Args:
        model_name (str): timm model name e.g: ``"resnet50"``. List of models can be found at
            `PyTorch Image Models <https://github.com/rwightman/pytorch-image-models>`_.
        pretrained (bool, optional): Imagenet pretrained. Default: ``False``.
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        drop_rate (float, optional): Dropout rate. Default: ``0.0``.
        drop_path_rate (float, optional): Drop path rate (model default if ``None``). Default: ``None``.
        drop_block_rate (float, optional): Drop block rate (model default if ``None``). Default: ``None``.
        global_pool (str, optional): Global pool type, one of (``"fast"``, ``"avg"``, ``"max"``, ``"avgmax"``, ``"avgmaxc"``). Model default if ``None``. Default: ``None``.
        bn_momentum (float, optional): BatchNorm momentum override (model default if ``None``). Default: ``None``.
        bn_eps (float, optional): BatchNorm epsilon override (model default if ``None``). Default: ``None``.

    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with the specified TIMM model.

    Resnet18 Example:

    .. testcode::

        from composer.models import composer_timm

        model = composer_timm(model_name='resnet18')  # creates a timm resnet18
    """
    try:
        import timm
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='timm', conda_package='timm>=0.5.4',
                                            conda_channel=None) from e
    model = timm.create_model(model_name=model_name,
                              pretrained=pretrained,
                              num_classes=num_classes,
                              drop_rate=drop_rate,
                              drop_path_rate=drop_path_rate,
                              drop_block_rate=drop_block_rate,
                              global_pool=global_pool,
                              bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

    composer_model = ComposerClassifier(module=model)
    return composer_model
