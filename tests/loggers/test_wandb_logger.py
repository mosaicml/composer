# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.loggers import WandBLoggerHparams
from composer.trainer import TrainerHparams


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
def test_wandb_logger(composer_trainer_hparams: TrainerHparams, world_size: int):
    pytest.importorskip("wandb", reason="wandb is an optional dependency")
    del world_size  # unused. Set via launcher script
    composer_trainer_hparams.loggers = [WandBLoggerHparams(log_artifacts=True)]
    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()
