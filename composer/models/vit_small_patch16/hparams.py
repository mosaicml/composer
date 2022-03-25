# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ViTSmallPatch16`."""

import textwrap
from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ["ViTSmallPatch16Hparams"]


@dataclass
class ViTSmallPatch16Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.ViTSmallPatch16`.

    Args:
        num_classes (int, optional): number of classes for the model. Default: ``1000``.
        image_size (int, optional): input image size. If you have rectangular images, make sure your image
         size is the maximum of the width and height. Default: ``224``.
        channels (int, optional): number of  image channels. Default: ``3``.
        dropout (float, optional): 0.0 - 1.0 dropout rate. Default: ``0``.
        embedding_dropout (float, optional): 0.0 - 1.0 embedding dropout rate. Default: ``0``.
    """
    num_classes: int = hp.optional("number of classes.  Needed for classification tasks", default=1000)
    image_size: int = hp.optional(
        "input image size. If you have rectangular images, make sure your image size is the maximum of the width and height",
        default=224)
    channels: int = hp.optional("number of  image channels", default=3)
    dropout: float = hp.optional("dropout rate", default=0.0)
    embedding_dropout: float = hp.optional("embedding dropout rate", default=0.0)

    def validate(self):
        try:
            import vit_pytorch  # type: ignore
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without vit support. To use vit with Composer, run `pip install mosaicml[vit]`
                if using pip or `pip install vit_pytorch>=0.27` if using Anaconda.""")) from e

    def initialize_object(self):
        from composer.models import ViTSmallPatch16
        return ViTSmallPatch16(num_classes=self.num_classes,
                               image_size=self.image_size,
                               channels=self.channels,
                               dropout=self.dropout,
                               embedding_dropout=self.embedding_dropout)
