# Copyright 2021 MosaicML. All Rights Reserved.

"""Train models with Composer speedup methods!"""

from composer.trainer import devices as devices
from composer.trainer.trainer import Trainer as Trainer
from composer.trainer.trainer_hparams import TrainerHparams as TrainerHparams

load = TrainerHparams.load

__all__ = ["Trainer"]
