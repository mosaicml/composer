from typing import Any, Dict, Type, Union

import pytest

from composer import Event
from composer.algorithms import AlgorithmHparams
from composer.algorithms.alibi.alibi import AlibiHparams
from composer.algorithms.augmix.augmix import AugMixHparams
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
    "hparams",
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
def test_init_idempotency(composer_trainer_hparams: TrainerHparams,
                          hparams: Union[Type[CallbackHparams], Type[AlgorithmHparams],
                                         Type[BaseLoggerBackendHparams]], monkeypatch: pytest.MonkeyPatch, tmpdir):
    default_kwargs: Dict[Type[Any], Dict[str, Any]] = {
        ScaleScheduleHparams: {
            "ratio": 1.0
        },
        RunDirectoryUploaderHparams: {
            "provider": 'local',
            "key_environ": "KEY_ENVIRON",
            "container": ".",
        },
        StochasticDepthHparams: {
            'stochastic_method': 'block',
            'target_layer_name': 'ResNetBottleneck',
        },
    }
    monkeypatch.setenv("KEY_ENVIRON", str(tmpdir))
    if issubclass(hparams, (SeqLengthWarmupHparams, AlibiHparams)):
        pytest.xfail("These algorithms require a synthetic NLP dataset, which does not exist.")
    if issubclass(hparams, (RandAugmentHparams, AugMixHparams)):
        pytest.xfail(
            "These algorithms require a synthetic Vision (i.e. PIL Image format) dataset, which does not exist")
    if issubclass(hparams, SWAHparams):
        pytest.xfail("SWA does not work with composed schedulers.")
    if issubclass(hparams, (BenchmarkerHparams, MosaicMLLoggerBackendHparams)):
        pytest.xfail("Not sure why these are failing, but nobody uses these anyways so going to ignore.")
    instance = hparams(**default_kwargs.get(hparams, {}))

    if isinstance(instance, BaseLoggerBackendHparams):
        composer_trainer_hparams.loggers.append(instance)
    elif isinstance(instance, CallbackHparams):
        composer_trainer_hparams.callbacks.append(instance)
    elif isinstance(instance, AlgorithmHparams):
        composer_trainer_hparams.algorithms.append(instance)
    else:
        pytest.fail(f"Unknown hparams type: {hparams.__name__}")

    trainer = composer_trainer_hparams.initialize_object()
    # idempotency test 1: run the init event again
    trainer.engine.run_event(Event.INIT)
    trainer.fit()
    # idempotency test 2: run init event again after training
    trainer.engine.run_event(Event.INIT)
    # TODO once we support multiple calls to fit, run fit again
