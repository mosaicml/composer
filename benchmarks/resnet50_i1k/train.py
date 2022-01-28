# Copyright 2021 MosaicML. All Rights Reserved.
from dataclasses import asdict, dataclass

import composer
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from composer.models.model_hparams import ModelHparams

from model import ResNet50


@dataclass
class ResNet50Hparams(ModelHparams):
    def initialize_object(self):
        return ResNet50(**asdict(self))

# hacky registration turn this into a decorator
composer.trainer.trainer_hparams.model_registry['resnet50'] = ResNet50Hparams

def main():
    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    trainer = Trainer.create_from_hparams(hparams=hparams)
    trainer.fit()


if __name__ == "__main__":
    main()
