# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.composer_yolox`."""

from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ['YoloXHparams']


@dataclass
class YoloXHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.composer_yolox`.
    Args:
        model_name (str): ``yolox-s``, ``yolox-m``, ``yolox-l`` or ``yolox-x``.
        num_classes (int, optional): The number of classes. Default: ``80``.
    """

    model_name: Optional[str] = hp.optional('yolox-s, yolox-m, yolox-l or yolox-x', default=None)
    num_classes: int = hp.optional('The number of classes.  Needed for classification tasks', default=80)

    def initialize_object(self):
        from composer.models.yolox.model import composer_yolox
        if self.model_name is None:
            raise ValueError('model_name must be specified')
        return composer_yolox(model_name=self.model_name, num_classes=self.num_classes)
