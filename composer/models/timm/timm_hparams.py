# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.composer_timm`."""

import textwrap
from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.models.model_hparams import ModelHparams
from composer.models.timm.model import composer_timm
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['TimmHparams']


@dataclass
class TimmHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.composer_timm`.

    Args:
        model_name (str): timm model name e.g: ``"resnet50"``. List of models can be found at
            `PyTorch Image Models <https://github.com/rwightman/pytorch-image-models>`_.
        pretrained (bool, optional): Imagenet pretrained. Default: ``False``.
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        drop_rate (float, optional): Dropout rate. Default: ``0.0``.
        drop_path_rate (float, optional): Drop path rate (model default if ``None``). Default: ``None``.
        drop_block_rate (float, optional): Drop block rate (model default if ``None``). Default: ``None``.
        global_pool (str, optional): Global pool type, one of (``"fast"``, ``"avg"``, ``"max"``, ``"avgmax"``, ``"avgmaxc"``). Model default if ``None``. Default: ``None``.
        bn_momentum (float, optional): BatchNorm momentum override (model default if not None). Default: ``None``.
        bn_eps (float, optional): BatchNorm epsilon override (model default if ``None``). Default: ``None``.
    """

    model_name: Optional[str] = hp.optional(
        textwrap.dedent("""\
        timm model name e.g: 'resnet50', list of models can be found at
        https://github.com/rwightman/pytorch-image-models"""),
        default=None,
    )
    pretrained: bool = hp.optional('imagenet pretrained', default=False)
    num_classes: int = hp.optional('The number of classes.  Needed for classification tasks', default=1000)
    drop_rate: float = hp.optional('dropout rate', default=0.0)
    drop_path_rate: Optional[float] = hp.optional('drop path rate (model default if None)', default=None)
    drop_block_rate: Optional[float] = hp.optional('drop block rate (model default if None)', default=None)
    global_pool: Optional[str] = hp.optional(
        'Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.', default=None)
    bn_momentum: Optional[float] = hp.optional('BatchNorm momentum override (model default if not None)', default=None)
    bn_eps: Optional[float] = hp.optional('BatchNorm epsilon override (model default if not None)', default=None)

    def validate(self):
        if self.model_name is None:
            try:
                import timm
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='timm',
                                                    conda_package='timm >=0.5.4',
                                                    conda_channel=None) from e
            raise ValueError(f'model must be one of {timm.models.list_models()}')

    def initialize_object(self):
        if self.model_name is None:
            raise ValueError('model_name must be specified')
        return composer_timm(model_name=self.model_name,
                             pretrained=self.pretrained,
                             num_classes=self.num_classes,
                             drop_rate=self.drop_rate,
                             drop_path_rate=self.drop_path_rate,
                             drop_block_rate=self.drop_block_rate,
                             global_pool=self.global_pool,
                             bn_momentum=self.bn_momentum,
                             bn_eps=self.bn_eps)
