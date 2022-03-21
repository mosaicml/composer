# Copyright 2021 MosaicML. All Rights Reserved.

"""Train models with flexible insertion of algorithms."""

from composer.trainer import devices as devices
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams

load = TrainerHparams.load

__all__ = ["Trainer", "TrainerHparams"]
