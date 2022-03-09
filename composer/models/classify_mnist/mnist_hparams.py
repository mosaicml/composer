# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ["MnistClassifierHparams"]


@dataclass
class MnistClassifierHparams(ModelHparams):
    """yahp Hparams interface for the simple mnist classifier.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: 10.
        initializers (List[Initializer], optional): Initializers
            for the model. ``None`` for no initialization.
            Default: ``None``.
    """

    def initialize_object(self):
        from composer.models import MNIST_Classifier
        return MNIST_Classifier(**asdict(self))
