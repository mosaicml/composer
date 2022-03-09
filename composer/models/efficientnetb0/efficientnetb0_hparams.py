# Copyright 2021 MosaicML. All Rights Reserved.

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.EfficientNetB0`."""

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ["EfficientNetB0Hparams"]


@dataclass
class EfficientNetB0Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.EfficientNetB0`.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``1000``.
        drop_connect_rate (float, optional): Probability of dropping a sample within a block before identity connection. Default: ``0.2``.
    """
    drop_connect_rate: float = hp.optional(
        doc="Probability of dropping a sample within a block before identity connection.",
        default=0.2,
    )

    def initialize_object(self):
        if self.num_classes is None:
            raise ValueError("EfficientNet requires num_classes to be specified.")

        from composer.models.efficientnetb0.model import EfficientNetB0
        return EfficientNetB0(num_classes=self.num_classes, drop_connect_rate=self.drop_connect_rate)
