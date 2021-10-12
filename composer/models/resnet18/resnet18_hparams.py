from dataclasses import dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class ResNet18Hparams(ModelHparams):

    def initialize_object(self):
        from composer.models import Resnet18
        return Resnet18(self)
