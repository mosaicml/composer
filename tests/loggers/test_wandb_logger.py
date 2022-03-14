# Copyright 2021 MosaicML. All Rights Reserved.

import pytest

from composer.loggers import WandBLoggerHparams
from composer.trainer import TrainerHparams


@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.timeout(10)
def test_wandb_logger(composer_trainer_hparams: TrainerHparams, world_size: int):
    pytest.importorskip("wandb", reason="wandb is an optional dependency")
    del world_size  # unused. Set via launcher script
    composer_trainer_hparams.loggers = [
        WandBLoggerHparams(log_artifacts=True, log_artifacts_every_n_batches=1, extra_init_params={"mode": "disabled"})
    ]
    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()
