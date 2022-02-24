# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import asdict, dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class MnistClassifierHparams(ModelHparams):
    """See :class:`MNIST_Classifier`"""

    def initialize_object(self):
        from composer.models import MNIST_Classifier
        return MNIST_Classifier(**asdict(self))
