# Copyright 2021 MosaicML. All Rights Reserved.

"""A wrapper around `timm.create_model() <https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model>`_
used to create :class:`.ComposerClassifier`."""

import textwrap
from typing import Optional

from composer.models.base import ComposerClassifier

__all__ = ["Timm"]


class Timm(ComposerClassifier):
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

    Resnet18 Example:

    .. testcode::

        from composer.models import Timm

        model = Timm(model_name='resnet18')  # creates a timm resnet18
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        drop_path_rate: Optional[float] = None,
        drop_block_rate: Optional[float] = None,
        global_pool: Optional[str] = None,
        bn_momentum: Optional[float] = None,
        bn_eps: Optional[float] = None,
    ) -> None:
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without timm support. To use timm with Composer, run `pip install mosaicml[timm]`
                if using pip or `pip install timm>=0.5.4` if using Anaconda.""")) from e

        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=drop_block_rate,
            global_pool=global_pool,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )
        super().__init__(module=model)
