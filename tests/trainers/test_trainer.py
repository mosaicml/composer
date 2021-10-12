from unittest.mock import patch

import pytest
from torch.optim import Adam

from composer.callbacks.lr_monitor import LRMonitor
from composer.core.logging.logger import Logger
from composer.datasets.hparams import DataloaderSpec
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.models.base import BaseMosaicModel
from composer.optim.optimizer_hparams import AdamHparams
from composer.optim.scheduler import ComposedScheduler, ExponentialLRHparams
from composer.trainer import Trainer, TrainerHparams


def test_trainer_init_all_defaults(dummy_dataloader_spec: DataloaderSpec, dummy_model: BaseMosaicModel):
    trainer = Trainer(model=dummy_model,
                      train_dataloader_spec=dummy_dataloader_spec,
                      eval_dataloader_spec=dummy_dataloader_spec,
                      max_epochs=10,
                      train_batch_size=32,
                      eval_batch_size=32)

    assert isinstance(trainer, Trainer)


def test_trainer_init_additional_args(dummy_dataloader_spec: DataloaderSpec, dummy_model: BaseMosaicModel):
    trainer = Trainer(
        model=dummy_model,
        train_dataloader_spec=dummy_dataloader_spec,
        eval_dataloader_spec=dummy_dataloader_spec,
        max_epochs=10,
        train_batch_size=32,
        eval_batch_size=32,
        optimizer_hparams=AdamHparams(),
        schedulers_hparams=[ExponentialLRHparams(gamma=0.1)],
        log_destinations=[TQDMLoggerBackend()],
        callbacks=(LRMonitor(),),
    )

    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.state.optimizers, Adam)

    assert isinstance(trainer.state.schedulers, ComposedScheduler)

    assert len(trainer.logger._log_destinations) == 1
    assert isinstance(trainer.logger._log_destinations[0], TQDMLoggerBackend)
    assert isinstance(trainer.logger, Logger)

    # log destination and lr monitor, logger destination callback must be first
    assert len(trainer.engine.callbacks) == 2
    assert isinstance(trainer.engine.callbacks[0], TQDMLoggerBackend)
    assert isinstance(trainer.engine.callbacks[1], LRMonitor)


def test_trainer_create_from_hparams(mosaic_trainer_hparams: TrainerHparams):
    trainer = Trainer.create_from_hparams(hparams=mosaic_trainer_hparams)
    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize('invalid_hparams', [])
def test_trainer_validation(mosaic_trainer_hparams: TrainerHparams, invalid_hparams):
    with patch.multiple(mosaic_trainer_hparams, **invalid_hparams), pytest.raises(ValueError):
        mosaic_trainer_hparams.validate()
