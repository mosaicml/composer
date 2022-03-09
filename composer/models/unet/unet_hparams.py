# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for
:class:`~composer.models.unet.unet.UNet`."""

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ["UnetHparams"]


@dataclass
class UnetHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for
    :class:`~composer.models.unet.unet.UNet`.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``3``.
    """

    def initialize_object(self):
        from composer.models.unet.unet import UNet
        return UNet(**asdict(self))
