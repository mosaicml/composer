# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.mnist_model`."""

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ['MnistClassifierHparams']


@dataclass
class MnistClassifierHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :func:`.mnist_model`.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: 10.
        initializers (List[Initializer], optional): Initializers for the model. ``None`` for no initialization. Default: ``None``.
    """

    def initialize_object(self):
        from composer.models import mnist_model
        return mnist_model(**asdict(self))
