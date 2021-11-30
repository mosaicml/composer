# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import patch

import pytest
import torch
import torch.distributed
from torch.optim import Adam

from composer.callbacks.lr_monitor import LRMonitor
from composer.core.logging.logger import Logger
from composer.core.precision import Precision
from composer.datasets.hparams import DataloaderSpec
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.models.base import BaseMosaicModel
from composer.optim.optimizer_hparams import AdamHparams
from composer.optim.scheduler import ComposedScheduler, ExponentialLRHparams
from composer.trainer import Trainer, TrainerHparams
from composer.trainer.devices.device_hparams import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from tests.utils.trainer_fit import get_total_loss, train_model


def test_trainer_init_all_defaults(dummy_train_dataloader_spec: DataloaderSpec,
                                   dummy_val_dataloader_spec: DataloaderSpec, dummy_model: BaseMosaicModel,
                                   dummy_train_batch_size: int, dummy_val_batch_size: int):
    trainer = Trainer(model=dummy_model,
                      train_dataloader_spec=dummy_train_dataloader_spec,
                      eval_dataloader_spec=dummy_val_dataloader_spec,
                      max_epochs=10,
                      train_batch_size=dummy_train_batch_size,
                      eval_batch_size=dummy_val_batch_size)

    assert isinstance(trainer, Trainer)


def test_trainer_init_additional_args(dummy_train_dataloader_spec: DataloaderSpec,
                                      dummy_val_dataloader_spec: DataloaderSpec, dummy_model: BaseMosaicModel,
                                      dummy_train_batch_size: int, dummy_val_batch_size: int):
    trainer = Trainer(
        model=dummy_model,
        train_dataloader_spec=dummy_train_dataloader_spec,
        eval_dataloader_spec=dummy_val_dataloader_spec,
        max_epochs=10,
        train_batch_size=dummy_train_batch_size,
        eval_batch_size=dummy_val_batch_size,
        optimizer_hparams=AdamHparams(),
        schedulers_hparams=[ExponentialLRHparams(gamma=0.1)],
        log_destinations=[TQDMLoggerBackend()],
        callbacks=(LRMonitor(),),
    )

    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.state.optimizers, Adam)

    assert isinstance(trainer.state.schedulers, ComposedScheduler)

    assert len(trainer.logger.backends) == 1
    assert isinstance(trainer.logger.backends[0], TQDMLoggerBackend)
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


@pytest.mark.filterwarnings("ignore:Deterministic mode is activated:UserWarning")
@pytest.mark.timeout(90)
def test_trainer_determinism(mosaic_trainer_hparams: TrainerHparams):
    mosaic_trainer_hparams.seed = 10
    mosaic_trainer_hparams.deterministic_mode = True
    mosaic_trainer_hparams.max_epochs = 2

    first_trainer = Trainer.create_from_hparams(mosaic_trainer_hparams)
    first_trainer.fit()
    first_model = first_trainer.state.model.module
    assert isinstance(first_model, BaseMosaicModel)
    assert first_trainer.state.train_dataloader is not None
    first_loss = get_total_loss(first_model, first_trainer.state.train_dataloader)

    # Second trainer must be created after fitting the first so that the
    # seeds get fully reset for the second training run
    second_trainer = Trainer.create_from_hparams(mosaic_trainer_hparams)
    second_trainer.fit()
    second_model = second_trainer.state.model.module
    assert isinstance(second_model, BaseMosaicModel)
    assert second_trainer.state.train_dataloader is not None
    second_loss = get_total_loss(second_model, second_trainer.state.train_dataloader)

    torch.testing.assert_allclose(second_loss, first_loss)


@pytest.mark.timeout(90)
@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize("device_hparams", [
    pytest.param(CPUDeviceHparams(), id="cpu"),
    pytest.param(GPUDeviceHparams(), id="gpu", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize("grad_accum", [
    pytest.param(1, id="ga1"),
    pytest.param(2, id="ga2"),
])
@pytest.mark.parametrize("precision", [
    pytest.param(Precision.FP32, id="fp32"),
    pytest.param(Precision.AMP, id="amp"),
])
def test_trainer_fit(mosaic_trainer_hparams: TrainerHparams, device_hparams: DeviceHparams, world_size: int,
                     grad_accum: int, precision: Precision):
    del world_size  # unused. Set via env vars
    mosaic_trainer_hparams.device = device_hparams
    mosaic_trainer_hparams.grad_accum = grad_accum
    mosaic_trainer_hparams.precision = precision

    # Not supported
    if precision == Precision.AMP and isinstance(device_hparams, CPUDeviceHparams):
        return

    train_model(mosaic_trainer_hparams, max_epochs=2, run_loss_check=True)
