# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.vit_small_patch16`."""

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['ViTSmallPatch16Hparams']


@dataclass
class ViTSmallPatch16Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.vit_small_batch16`.

    Args:
        num_classes (int, optional): number of classes for the model. Default: ``1000``.
        image_size (int, optional): input image size. If you have rectangular images, make sure your image
         size is the maximum of the width and height. Default: ``224``.
        channels (int, optional): number of  image channels. Default: ``3``.
        dropout (float, optional): 0.0 - 1.0 dropout rate. Default: ``0``.
        embedding_dropout (float, optional): 0.0 - 1.0 embedding dropout rate. Default: ``0``.
    """
    num_classes: int = hp.optional('number of classes.  Needed for classification tasks', default=1000)
    image_size: int = hp.optional(
        'input image size. If you have rectangular images, make sure your image size is the maximum of the width and height',
        default=224)
    channels: int = hp.optional('number of  image channels', default=3)
    dropout: float = hp.optional('dropout rate', default=0.0)
    embedding_dropout: float = hp.optional('embedding dropout rate', default=0.0)

    def validate(self):
        try:
            import vit_pytorch
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='vit', conda_package='vit_pytorch>=0.27') from e
        del vit_pytorch  # unused

    def initialize_object(self):
        from composer.models import vit_small_patch16
        return vit_small_patch16(num_classes=self.num_classes,
                                 image_size=self.image_size,
                                 channels=self.channels,
                                 dropout=self.dropout,
                                 embedding_dropout=self.embedding_dropout)
