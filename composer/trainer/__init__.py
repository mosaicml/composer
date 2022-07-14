# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Train models with flexible insertion of algorithms."""

from composer.trainer import devices as devices
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_tpu import TrainerTPU

__all__ = ['Trainer', 'TrainerTPU']

