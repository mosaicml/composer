# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import patch

import pytest
import torch
import torch.distributed
from torch.optim import Adam

from composer.callbacks.lr_monitor import LRMonitor
from composer.core.logging.logger import Logger
from composer.core.precision import Precision
from composer.core.types import DataLoader
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.models.base import BaseMosaicModel
from composer.optim.optimizer_hparams import AdamHparams
from composer.optim.scheduler import ComposedScheduler, ExponentialLRHparams
from composer.trainer import Trainer, TrainerHparams
from composer.trainer.devices.device_hparams import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from tests.utils.trainer_fit import get_total_loss, train_model


def test_trainer_init_all_defaults(dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader,
                                   dummy_model: BaseMosaicModel):
    trainer = Trainer(model=dummy_model,
                      train_dataloader=dummy_train_dataloader,
                      eval_dataloader=dummy_val_dataloader,
                      max_duration="10ep")

    assert isinstance(trainer, Trainer)


def test_trainer_init_additional_args(dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader,
                                      dummy_model: BaseMosaicModel):
    trainer = Trainer(
        model=dummy_model,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
        max_duration="10ep",
        optimizer_hparams=AdamHparams(),
        schedulers_hparams=[ExponentialLRHparams(gamma=0.1)],
        log_destinations=[TQDMLoggerBackend()],
        callbacks=(LRMonitor(),),
    )

    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.state.optimizers[0], Adam)

    assert isinstance(trainer.state.schedulers[0], ComposedScheduler)

    assert len(trainer.logger.backends) == 1
    assert isinstance(trainer.logger.backends[0], TQDMLoggerBackend)
    assert isinstance(trainer.logger, Logger)

    # log destination and lr monitor, logger destination callback must be first
    assert len(trainer.state.callbacks) == 2
    assert isinstance(trainer.state.callbacks[0], TQDMLoggerBackend)
    assert isinstance(trainer.state.callbacks[1], LRMonitor)


def test_trainer_create_from_hparams(mosaic_trainer_hparams: TrainerHparams):
    trainer = Trainer.create_from_hparams(hparams=mosaic_trainer_hparams)
    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize('invalid_hparams', [])
def test_trainer_validation(mosaic_trainer_hparams: TrainerHparams, invalid_hparams):
    with patch.multiple(mosaic_trainer_hparams, **invalid_hparams), pytest.raises(ValueError):
        mosaic_trainer_hparams.validate()


@pytest.mark.timeout(90)
@pytest.mark.parametrize("device", [CPUDeviceHparams(), pytest.param(GPUDeviceHparams(), marks=pytest.mark.gpu)])
def test_trainer_determinism(mosaic_trainer_hparams: TrainerHparams, device: DeviceHparams):
    mosaic_trainer_hparams.seed = 10
    mosaic_trainer_hparams.device = device
    mosaic_trainer_hparams.max_duration = "2ep"

    first_trainer = Trainer.create_from_hparams(mosaic_trainer_hparams)
    first_trainer.fit()
    first_model = first_trainer.state.model.module
    assert isinstance(first_model, BaseMosaicModel)
    assert first_trainer.state.train_dataloader is not None
    first_loss = get_total_loss(first_model, first_trainer.state.train_dataloader, first_trainer.device)

    # Second trainer must be created after fitting the first so that the
    # seeds get fully reset for the second training run
    second_trainer = Trainer.create_from_hparams(mosaic_trainer_hparams)
    second_trainer.fit()
    second_model = second_trainer.state.model.module
    assert isinstance(second_model, BaseMosaicModel)
    assert second_trainer.state.train_dataloader is not None
    second_loss = get_total_loss(second_model, second_trainer.state.train_dataloader, second_trainer.device)

    torch.testing.assert_allclose(second_loss, first_loss)


@pytest.mark.timeout(90)
@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize("device_hparams,precision", [
    pytest.param(CPUDeviceHparams(), Precision.FP32, id="cpu"),
    pytest.param(GPUDeviceHparams(), Precision.FP32, id="gpu-fp32", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), Precision.AMP, id="gpu-amp", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize("grad_accum", [
    pytest.param(1, id="ga1"),
    pytest.param(2, id="ga2"),
])
def test_trainer_fit(mosaic_trainer_hparams: TrainerHparams, device_hparams: DeviceHparams, world_size: int,
                     grad_accum: int, precision: Precision):
    del world_size  # unused. Set via env vars
    mosaic_trainer_hparams.device = device_hparams
    mosaic_trainer_hparams.grad_accum = grad_accum
    mosaic_trainer_hparams.precision = precision

    train_model(mosaic_trainer_hparams, max_epochs=2, run_loss_check=True)
