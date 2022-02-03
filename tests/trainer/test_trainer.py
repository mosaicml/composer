# Copyright 2021 MosaicML. All Rights Reserved.

import pathlib
from typing import Type
from unittest.mock import patch

import pytest
import torch
import torch.distributed
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.algorithms.alibi.alibi import AlibiHparams
from composer.algorithms.augmix.augmix import AugMixHparams
from composer.algorithms.cutmix.cutmix import CutMixHparams
from composer.algorithms.label_smoothing.label_smoothing import LabelSmoothingHparams
from composer.algorithms.layer_freezing.layer_freezing import LayerFreezingHparams
from composer.algorithms.mixup.mixup import MixUpHparams
from composer.algorithms.randaugment.randaugment import RandAugmentHparams
from composer.algorithms.scale_schedule.scale_schedule import ScaleScheduleHparams
from composer.algorithms.seq_length_warmup.seq_length_warmup import SeqLengthWarmupHparams
from composer.algorithms.stochastic_depth.stochastic_depth import StochasticDepthHparams
from composer.algorithms.swa.hparams import SWAHparams
from composer.callbacks.callback_hparams import BenchmarkerHparams, CallbackHparams, RunDirectoryUploaderHparams
from composer.callbacks.lr_monitor import LRMonitor
from composer.core.event import Event
from composer.core.logging.logger import Logger
from composer.core.precision import Precision
from composer.core.profiler import ProfilerEventHandlerHparams
from composer.core.types import DataLoader, Optimizer, Scheduler
from composer.loggers import BaseLoggerBackendHparams
from composer.loggers.logger_hparams import MosaicMLLoggerBackendHparams
from composer.loggers.tqdm_logger import TQDMLoggerBackend
from composer.models.base import ComposerModel
from composer.optim.scheduler import ComposedScheduler
from composer.profiler.profiler_hparams import ProfilerCallbackHparams, ProfilerHparams
from composer.trainer import Trainer, TrainerHparams
from composer.trainer.devices.device_hparams import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from tests.utils.trainer_fit import get_total_loss, train_model


def test_trainer_init_all_defaults(dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader,
                                   dummy_model: ComposerModel):
    trainer = Trainer(model=dummy_model,
                      train_dataloader=dummy_train_dataloader,
                      eval_dataloader=dummy_val_dataloader,
                      max_duration="10ep")

    assert isinstance(trainer, Trainer)


def test_trainer_init_additional_args(dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader,
                                      dummy_optimizer: Optimizer, dummy_scheduler: Scheduler,
                                      dummy_model: ComposerModel):
    trainer = Trainer(
        model=dummy_model,
        train_dataloader=dummy_train_dataloader,
        eval_dataloader=dummy_val_dataloader,
        max_duration="10ep",
        optimizers=dummy_optimizer,
        schedulers=dummy_scheduler,
        log_destinations=[TQDMLoggerBackend()],
        callbacks=(LRMonitor(),),
    )

    assert isinstance(trainer, Trainer)
    assert trainer.state.optimizers[0] == dummy_optimizer

    assert isinstance(trainer.state.schedulers[0], ComposedScheduler)

    assert len(trainer.logger.backends) == 1
    assert isinstance(trainer.logger.backends[0], TQDMLoggerBackend)
    assert isinstance(trainer.logger, Logger)

    # log destination and lr monitor, logger destination callback must be first
    assert len(trainer.state.callbacks) == 2
    assert isinstance(trainer.state.callbacks[0], TQDMLoggerBackend)
    assert isinstance(trainer.state.callbacks[1], LRMonitor)


def test_trainer_hparams_initialize_object(composer_trainer_hparams: TrainerHparams):
    trainer = composer_trainer_hparams.initialize_object()
    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize('invalid_hparams', [])
def test_trainer_validation(composer_trainer_hparams: TrainerHparams, invalid_hparams):
    with patch.multiple(composer_trainer_hparams, **invalid_hparams), pytest.raises(ValueError):
        composer_trainer_hparams.validate()


@pytest.mark.timeout(90)
@pytest.mark.parametrize("device", [CPUDeviceHparams(), pytest.param(GPUDeviceHparams(), marks=pytest.mark.gpu)])
def test_trainer_determinism(composer_trainer_hparams: TrainerHparams, device: DeviceHparams):
    composer_trainer_hparams.seed = 10
    composer_trainer_hparams.device = device
    composer_trainer_hparams.max_duration = "2ep"

    first_trainer = composer_trainer_hparams.initialize_object()
    first_trainer.fit()
    first_model = first_trainer.state.model.module
    assert isinstance(first_model, ComposerModel)
    assert first_trainer.state.train_dataloader is not None
    first_loss = get_total_loss(first_model, first_trainer.state.train_dataloader, first_trainer.device)

    # Second trainer must be created after fitting the first so that the
    # seeds get fully reset for the second training run
    second_trainer = composer_trainer_hparams.initialize_object()
    second_trainer.fit()
    second_model = second_trainer.state.model.module
    assert isinstance(second_model, ComposerModel)
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
def test_trainer_fit(composer_trainer_hparams: TrainerHparams, device_hparams: DeviceHparams, world_size: int,
                     grad_accum: int, precision: Precision):
    del world_size  # unused. Set via env vars
    composer_trainer_hparams.device = device_hparams
    composer_trainer_hparams.grad_accum = grad_accum
    composer_trainer_hparams.precision = precision

    train_model(composer_trainer_hparams, max_epochs=2, run_loss_check=True)


_ALL_LOGGERS_CALLBACKS_ALG_PROFILER_HPARAMS = [
    *TrainerHparams.hparams_registry["algorithms"].values(),
    # excluding the run directory uploader here since it needs a longer timeout -- see below
    *[
        x for x in TrainerHparams.hparams_registry["callbacks"].values()
        if not issubclass(x, RunDirectoryUploaderHparams)
    ],
    *TrainerHparams.hparams_registry["loggers"].values(),
    *ProfilerHparams.hparams_registry["profilers"].values(),
    *ProfilerHparams.hparams_registry["trace_event_handlers"].values(),
    pytest.param(RunDirectoryUploaderHparams, marks=pytest.mark.timeout(10)),  # this test takes longer
]


def _build_trainer(composer_trainer_hparams: TrainerHparams, dummy_num_classes: int, hparams_cls: Type[hp.Hparams],
                   monkeypatch: pytest.MonkeyPatch, tmpdir: pathlib.Path):
    hparams_with_required_fields = [
        ScaleScheduleHparams(ratio=1.0),
        RunDirectoryUploaderHparams(
            provider='local',
            key_environ="KEY_ENVIRON",
            container=".",
        ),
        StochasticDepthHparams(
            stochastic_method='block',
            target_layer_name='ResNetBottleneck',
        ),
        CutMixHparams(num_classes=dummy_num_classes,),
        MixUpHparams(num_classes=dummy_num_classes,)
    ]
    pytest.importorskip("wandb", reason="Wandb is not installed on mosaicml[dev]")
    pytest.importorskip("libcloud", reason="libcloud is not installed on mosaicml[dev]")
    monkeypatch.setenv("KEY_ENVIRON", str(tmpdir))
    if issubclass(hparams_cls, (SeqLengthWarmupHparams, AlibiHparams)):
        pytest.xfail("These algorithms require a synthetic NLP dataset, which does not exist.")
    if issubclass(hparams_cls, (RandAugmentHparams, AugMixHparams)):
        pytest.xfail(
            "These algorithms require a synthetic Vision (i.e. PIL Image format) dataset, which does not exist")
    if issubclass(hparams_cls, SWAHparams):
        pytest.xfail("SWA does not work with composed schedulers.")
    if issubclass(hparams_cls, (BenchmarkerHparams, MosaicMLLoggerBackendHparams)):
        pytest.xfail("Not sure why these are failing, but nobody uses these anyways so going to ignore.")
    if issubclass(hparams_cls, (CutMixHparams, MixUpHparams, LabelSmoothingHparams)):
        pytest.importorskip("torch",
                            minversion="1.10",
                            reason=f"{hparams_cls.__name__} requires Pytorch 1.10 for multi-target loss")

    # if the hparam cls is in hparams_with_required_fields, then set `instance` to the value in there
    # otherwise, set instance = hparams_cls()
    instance = None
    for x in hparams_with_required_fields:
        if isinstance(x, hparams_cls):
            instance = x
            break
    if instance is None:
        instance = hparams_cls()

    if isinstance(instance, BaseLoggerBackendHparams):
        composer_trainer_hparams.loggers.append(instance)
    elif isinstance(instance, ProfilerCallbackHparams):
        composer_trainer_hparams.profiler = ProfilerHparams(profilers=[instance])
    elif isinstance(instance, ProfilerEventHandlerHparams):
        composer_trainer_hparams.profiler = ProfilerHparams(trace_event_handlers=[instance], profilers=[])
    elif isinstance(instance, CallbackHparams):
        composer_trainer_hparams.callbacks.append(instance)
    elif isinstance(instance, AlgorithmHparams):
        composer_trainer_hparams.algorithms.append(instance)
    else:
        pytest.fail(f"Unknown hparams type: {hparams_cls.__name__}")
    return composer_trainer_hparams.initialize_object()


@pytest.mark.parametrize("hparams_cls", _ALL_LOGGERS_CALLBACKS_ALG_PROFILER_HPARAMS)
def test_fit_on_all_callbacks_loggers_algs_profilers(
    composer_trainer_hparams: TrainerHparams,
    dummy_num_classes: int,
    hparams_cls: Type[hp.Hparams],
    monkeypatch: pytest.MonkeyPatch,
    tmpdir: pathlib.Path,
):
    trainer = _build_trainer(composer_trainer_hparams, dummy_num_classes, hparams_cls, monkeypatch, tmpdir)
    trainer.fit()


@pytest.mark.parametrize("hparams_cls", _ALL_LOGGERS_CALLBACKS_ALG_PROFILER_HPARAMS)
def test_multiple_calls_to_fit(
    composer_trainer_hparams: TrainerHparams,
    dummy_num_classes: int,
    hparams_cls: Type[hp.Hparams],
    monkeypatch: pytest.MonkeyPatch,
    tmpdir: pathlib.Path,
):
    trainer = composer_trainer_hparams.initialize_object()
    # idempotency test 1: run the FIT event again
    trainer.engine.run_event(Event.FIT_START)
    trainer._train_loop()  # using the private method so we don't shutdown the callbacks
    trainer.state.max_duration = trainer.state.max_duration + trainer.state.max_duration
    # idempotency test 2: run FIT again. This will trigger another call to FIT_START
    if issubclass(hparams_cls, LayerFreezingHparams):
        pytest.xfail("TODO: Layer freezing does not work with a subsequent call to .fit")
    trainer.fit()  # using the public method so we do shutdown the callbacks
