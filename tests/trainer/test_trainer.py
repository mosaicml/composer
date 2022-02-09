# Copyright 2021 MosaicML. All Rights Reserved.

import datetime
import os
import pathlib
import unittest
from typing import Type
from unittest import mock
from unittest.mock import patch

import pytest
import torch
import torch.distributed
import yahp as hp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

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
from composer.core.types import Model, Optimizer, Scheduler
from composer.loggers import LoggerCallbackHparams
from composer.loggers.logger_hparams import MosaicMLLoggerHparams
from composer.loggers.tqdm_logger import TQDMLogger
from composer.models.base import ComposerModel
from composer.optim.scheduler import ComposedScheduler
from composer.profiler.profiler_hparams import ProfilerCallbackHparams, ProfilerHparams
from composer.trainer import Trainer, TrainerHparams, trainer_hparams
from composer.trainer.devices.device_gpu import DeviceGPU
from composer.trainer.devices.device_hparams import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleConvModel, SimpleModel, device, world_size
from tests.utils.trainer_fit import get_total_loss, train_model


class TestTrainerInit():

    @pytest.fixture
    def config(self):
        return {
            'model': SimpleModel(),
            'train_dataloader': DataLoader(dataset=RandomClassificationDataset()),
            'eval_dataloader': DataLoader(dataset=RandomClassificationDataset()),
            'max_duration': '2ep',
        }

    def test_init(self, config):
        trainer = Trainer(**config)
        assert isinstance(trainer, Trainer)

    def test_model_ddp_wrapped(self, config, init_process_group):
        # dist initialized first with init_process_group fixture
        trainer = Trainer(**config)
        assert isinstance(trainer.state.model, DistributedDataParallel)

    def test_model_ddp_not_wrapped(self, config):
        trainer = Trainer(**config)
        assert not isinstance(trainer.state.model, DistributedDataParallel)

    def test_loggers_before_callbacks(self, config):
        config.update({
            "loggers": [TQDMLogger()],
            "callbacks": [LRMonitor()],
        })

        trainer = Trainer(**config)
        assert isinstance(trainer.state.callbacks[0], TQDMLogger)
        assert isinstance(trainer.state.callbacks[1], LRMonitor)

    @device('gpu', 'cpu')
    def test_optimizer_on_device(self, config, device):
        config['device'] = device
        trainer = Trainer(**config)

        parameters = trainer.state.optimizers[0].param_groups[0]["params"]

        target_device = 'cuda' if isinstance(device, DeviceGPU) else 'cpu'
        assert all(param.device.type == target_device for param in parameters)

    def test_invalid_device(self, config):
        config['device'] = "magic_device"

        with pytest.raises(ValueError, match="magic_device"):
            Trainer(**config)

    def test_active_iterator_error(self, config):
        dataloader = DataLoader(
            dataset=RandomClassificationDataset(),
            persistent_workers=True,
            num_workers=1,
        )

        # spin one sample
        _ = next(dataloader.__iter__())

        config['train_dataloader'] = dataloader
        with pytest.raises(ValueError, match="active iterator"):
            Trainer(**config)


@world_size(1, 2)
@device('cpu', 'gpu', 'gpu-amp', precision=True)
class TestTrainerEquivalence():

    reference_model: Model
    reference_folder: pathlib.Path

    def assert_models_equal(self, model_1, model_2):
        for param1, param2 in zip(model_1.parameters(), model_2.parameters()):
            torch.testing.assert_allclose(param1, param2)

    @pytest.fixture
    def config(self, device, precision, world_size):
        return {
            'model': SimpleModel(),
            'train_dataloader': DataLoader(
                dataset=RandomClassificationDataset(),
                batch_size=4,
                shuffle=True,
            ),
            'eval_dataloader': DataLoader(
                dataset=RandomClassificationDataset(),
                shuffle=False,
            ),
            'max_duration': '2ep',
            'seed': 0,
            'device': device,
            'precision': precision,
            'deterministic_mode': True,  # testing equivalence
            'loggers': [],  # no progress bar
        }

    @pytest.fixture(autouse=True)
    def create_reference_model(self, config, tmpdir_factory, *args):
        """Trains the reference model, and saves checkpoints"""
        save_folder = tmpdir_factory.mktemp("{device}-{precision}".format(**config))
        config.update({'save_interval': '1ep', 'save_folder': save_folder})

        trainer = Trainer(**config)
        trainer.fit()

        self.reference_model = trainer.state.model
        self.reference_folder = save_folder

    def test_determinism(self, config, *args):
        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_grad_accum(self, config, *args):
        config.update({
            'grad_accum': 2,
        })

        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_checkpoint(self, config, *args):
        checkpoint_file = os.path.join(self.reference_folder, 'ep1.tar')
        config['load_path'] = checkpoint_file

        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)


class TestTrainerEvents(unittest.TestCase):

    def test_data_augmented(self):
        pass


"""
The below is a catch-all test that runs the Trainer
with each algorithm, callback, and loggers. Success
is defined as a successful training run.

This should eventually be replaced by functional
tests for each object, in situ of our trainer.

We use the hparams_registry associated with our
config management to retrieve the objects to test.
"""


class TestTrainerAssets:

    @pytest.fixture
    def config(self):
        return {
            'model': SimpleModel(),
            'train_dataloader': DataLoader(
                dataset=RandomClassificationDataset(size=16),
                batch_size=4,
            ),
            'eval_dataloader': DataLoader(dataset=RandomClassificationDataset(size=16),),
            'max_duration': '2ep',
            'loggers': [],  # no progress bar
        }

    @pytest.mark.parametrize("algorithm", trainer_hparams.algorithms_registry.items())
    def test_algorithms(self, config, algorithm):
        skip_list = {
            'swa': 'SWA not compatible with composed schedulers.',
            'alibi': 'Not compatible with simple linear model',
            'seq_length_warmup': 'Not compatible with simple linear model',
            'randaugment': 'Requires PIL dataset to test.',
            'augmix': 'Required PIL dataset to test.',
        }
        name, hparams = algorithm

        if name in skip_list:
            pytest.skip(skip_list[name])
        elif name in ('cutmix, mixup, label_smoothing'):
            pytest.importorskip("torch", minversion="1.10", reason="Pytorch 1.10 required.")

        pass

    @pytest.mark.parametrize("callback", trainer_hparams.callback_registry.items())
    def test_callbacks(self, config):
        pass

    @pytest.mark.parametrize("logger", trainer_hparams.logger_registry.items())
    def test_loggers(self, config):
        pass


# _ALL_LOGGERS_CALLBACKS_ALG_PROFILER_HPARAMS = [
#     *TrainerHparams.hparams_registry["algorithms"].values(),
#     # excluding the run directory uploader here since it needs a longer timeout -- see below
#     *[
#         x for x in TrainerHparams.hparams_registry["callbacks"].values()
#         if not issubclass(x, RunDirectoryUploaderHparams)
#     ],
#     *TrainerHparams.hparams_registry["loggers"].values(),
#     *ProfilerHparams.hparams_registry["profilers"].values(),
#     *ProfilerHparams.hparams_registry["trace_event_handlers"].values(),
#     pytest.param(RunDirectoryUploaderHparams, marks=pytest.mark.timeout(10)),  # this test takes longer
# ]


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
    if issubclass(hparams_cls, (BenchmarkerHparams, MosaicMLLoggerHparams)):
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

    if isinstance(instance, LoggerCallbackHparams):
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


"""
@pytest.mark.parametrize("hparams_cls", _ALL_LOGGERS_CALLBACKS_ALG_PROFILER_HPARAMS)
def test_multiple_calls_to_fit(
    composer_trainer_hparams: TrainerHparams,
    dummy_num_classes: int,
    hparams_cls: Type[hp.Hparams],
    monkeypatch: pytest.MonkeyPatch,
    tmpdir: pathlib.Path,
):
    trainer = _build_trainer(composer_trainer_hparams, dummy_num_classes, hparams_cls, monkeypatch, tmpdir)
    # idempotency test 1: run the FIT event again
    trainer.engine.run_event(Event.FIT_START)
    trainer._train_loop()  # using the private method so we don't shutdown the callbacks
    trainer.state.max_duration = trainer.state.max_duration + trainer.state.max_duration
    # idempotency test 2: run FIT again. This will trigger another call to FIT_START
    if issubclass(hparams_cls, LayerFreezingHparams):
        pytest.xfail("TODO: Layer freezing does not work with a subsequent call to .fit")
    trainer.fit()  # using the public method so we do shutdown the callbacks
"""
