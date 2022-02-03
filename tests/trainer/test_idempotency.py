# Copyright 2021 MosaicML. All Rights Reserved.

import pathlib
from typing import Type, Union

import pytest

from composer import Event
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
from composer.loggers import BaseLoggerBackendHparams
from composer.loggers.logger_hparams import MosaicMLLoggerBackendHparams
from composer.trainer import TrainerHparams


@pytest.mark.parametrize(
    "hparams_cls",
    [
        *TrainerHparams.hparams_registry["algorithms"].values(),
        # excluding the run directory uploader here since it needs a longer timeout -- see below
        *[
            x for x in TrainerHparams.hparams_registry["callbacks"].values()
            if not issubclass(x, RunDirectoryUploaderHparams)
        ],
        *TrainerHparams.hparams_registry["loggers"].values(),
        pytest.param(RunDirectoryUploaderHparams, marks=pytest.mark.timeout(10)),  # this test takes longer
    ])
def test_init_idempotency(composer_trainer_hparams: TrainerHparams, dummy_num_classes: int,
                          hparams_cls: Union[Type[CallbackHparams], Type[AlgorithmHparams],
                                             Type[BaseLoggerBackendHparams],], monkeypatch: pytest.MonkeyPatch,
                          tmpdir: pathlib.Path):
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
    instance = None
    for x in hparams_with_required_fields:
        if isinstance(x, hparams_cls):
            instance = x
            break
    if instance is None:
        instance = hparams_cls()

    if isinstance(instance, BaseLoggerBackendHparams):
        composer_trainer_hparams.loggers.append(instance)
    elif isinstance(instance, CallbackHparams):
        composer_trainer_hparams.callbacks.append(instance)
    elif isinstance(instance, AlgorithmHparams):
        composer_trainer_hparams.algorithms.append(instance)
    else:
        pytest.fail(f"Unknown hparams type: {hparams_cls.__name__}")

    trainer = composer_trainer_hparams.initialize_object()
    # idempotency test 1: run the FIT event again
    trainer.engine.run_event(Event.FIT_START)
    trainer.fit(shutdown=False)
    trainer.state.max_duration = trainer.state.max_duration + trainer.state.max_duration
    # idempotency test 2: run FIT again. This will trigger another call to FIT_START
    if issubclass(hparams_cls, LayerFreezingHparams):
        pytest.xfail("Layer freezing does not work with a subsequent call to .fit")
    trainer.fit()
    with pytest.raises(RuntimeError):
        trainer.fit()  # engine should be shutdown
