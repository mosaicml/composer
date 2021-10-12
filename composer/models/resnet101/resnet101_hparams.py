from dataclasses import dataclass

from composer.models.model_hparams import ModelHparams


@dataclass
class ResNet101Hparams(ModelHparams):

    def initialize_object(self):
        from composer.models import ResNet101
        return ResNet101(self)
