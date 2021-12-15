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
from tests.test_state import assert_state_equivalent
from tests.utils.trainer_fit import get_total_loss, train_model


def test_trainer_init_all_defaults(dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader,
                                   dummy_model: BaseMosaicModel):
    trainer = Trainer(model=dummy_model,
                      train_dataloader=dummy_train_dataloader,
                      eval_dataloader=dummy_val_dataloader,
                      max_epochs=10)

    assert isinstance(trainer, Trainer)


def test_trainer_init_additional_args(dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader,
                                      dummy_model: BaseMosaicModel):
    trainer = Trainer(
        model=dummy_model,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
        max_epochs=10,
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


def test_partial_train(mosaic_trainer_hparams: TrainerHparams):
    # Assert that following calls produce equivalent state:
    # 1. .fit() (train until end)
    # 2. for _ in range(max_epochs): .fit(num_epochs=1)
    # 3. for _ in range(max_peochs): for _ in range(steps_per_epoch): .fit(num_bathces=1)
    assert mosaic_trainer_hparams.max_epochs > 1
    assert mosaic_trainer_hparams.train_subset_num_batches is not None
    assert mosaic_trainer_hparams.train_subset_num_batches > 1
    mosaic_trainer_hparams.train_dataset.shuffle = False
    mosaic_trainer_hparams.val_dataset.shuffle = False

    trainer_1 = mosaic_trainer_hparams.initialize_object()
    model_state = trainer_1.model.state_dict()
    trainer_1.fit()
    assert trainer_1.engine.closed

    with pytest.raises(RuntimeError):
        # cannot over-train
        trainer_1.fit()

    trainer_2 = mosaic_trainer_hparams.initialize_object()
    trainer_2.model.load_state_dict(model_state)
    for _ in range(mosaic_trainer_hparams.max_epochs):
        trainer_2.fit(num_epochs=1)
    assert trainer_2.engine.closed

    trainer_3 = mosaic_trainer_hparams.initialize_object()
    trainer_3.model.load_state_dict(model_state)
    for _ in range(mosaic_trainer_hparams.max_epochs):
        for i in range(mosaic_trainer_hparams.train_subset_num_batches):
            trainer_3.fit(num_batches=1)
            if i == 0:
                with pytest.raises(ValueError):
                    trainer_3.fit(num_epochs=1)  # cannot fit with num_epochs when in the middle of a batch
    assert trainer_3.engine.closed

    trainer_4 = mosaic_trainer_hparams.initialize_object()
    trainer_4.model.load_state_dict(model_state)
    trainer_4.fit(num_epochs=1)
    trainer_4.fit()
    assert trainer_4.engine.closed

    trainer_5 = mosaic_trainer_hparams.initialize_object()
    trainer_5.model.load_state_dict(model_state)
    trainer_5.fit(num_batches=mosaic_trainer_hparams.train_subset_num_batches)
    trainer_5.fit(num_epochs=mosaic_trainer_hparams.max_epochs - 1)
    assert trainer_5.engine.closed

    assert_state_equivalent(trainer_1.state, trainer_2.state)
    assert_state_equivalent(trainer_1.state, trainer_3.state)
    assert_state_equivalent(trainer_1.state, trainer_4.state)
    assert_state_equivalent(trainer_1.state, trainer_5.state)
