# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from composer.algorithms import LayerFreezing
from composer.algorithms.cutout.cutout import CutOut
from composer.callbacks import LRMonitor
from composer.callbacks.run_directory_uploader import RunDirectoryUploader
from composer.core.callback import Callback
from composer.core.time import Time, TimeUnit
from composer.core.types import Model
from composer.loggers import FileLogger, TQDMLogger, WandBLogger
from composer.trainer import Trainer
from composer.trainer.trainer_hparams import algorithms_registry, callback_registry, logger_registry
from composer.utils import dist
from tests.common import (RandomClassificationDataset, RandomImageDataset, SimpleConvModel, SimpleModel, device,
                          world_size)


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

    def test_model_ddp_wrapped(self, config):
        trainer = Trainer(**config)
        should_be_ddp_wrapped = dist.get_world_size() > 1 and "deepspeed_config" not in config
        assert isinstance(trainer.state.model, DistributedDataParallel) == should_be_ddp_wrapped

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

        target_device = 'cuda' if device == 'gpu' else 'cpu'
        assert all(param.device.type == target_device for param in parameters)

    def test_invalid_device(self, config):
        config['device'] = "magic_device"

        with pytest.raises(ValueError, match="magic_device"):
            Trainer(**config)

    @pytest.mark.timeout(5.0)
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

    def test_init_with_integers(self, config, tmpdir):
        config.update({
            'max_duration': 1,
            'save_interval': 10,
            'save_folder': tmpdir,
        })

        trainer = Trainer(**config)
        assert trainer.state.max_duration == "1ep"
        assert trainer._checkpoint_saver is not None and \
            trainer._checkpoint_saver._save_interval == "10ep"

    def test_init_with_max_duration_in_batches(self, config):
        config["max_duration"] = '1ba'
        trainer = Trainer(**config)
        assert trainer.state.max_duration.to_timestring() == "1ba"


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
        """Trains the reference model, and saves checkpoints."""
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

    def test_max_duration(self, config, *args):
        max_duration = Time.from_timestring(config['max_duration'])
        assert max_duration.unit == TimeUnit.EPOCH
        max_duration_in_batches = Time(len(config['train_dataloader']) * int(max_duration.value), TimeUnit.BATCH)
        config['max_duration'] = max_duration_in_batches.to_timestring()
        trainer = Trainer(**config)
        assert trainer.state.max_duration.unit == TimeUnit.BATCH
        trainer.fit()
        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_checkpoint(self, config, *args):
        # load from epoch 1 checkpoint and finish training
        checkpoint_file = os.path.join(self.reference_folder, 'ep1.pt')
        config['load_path'] = checkpoint_file

        trainer = Trainer(**config)
        assert trainer.state.timer.epoch == "1ep"  # ensure checkpoint state loaded
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_model_init(self, config, *args):
        # as a control test, we reinitialize the model weights, and
        # expect the resulting trained model to differe from the reference.
        config['model'] = SimpleModel()

        trainer = Trainer(**config)
        trainer.fit()

        with pytest.raises(AssertionError):
            self.assert_models_equal(trainer.state.model, self.reference_model)


class AssertDataAugmented(Callback):
    """Helper callback that asserts test whether the augmented batch was passed to the model during the forward pass.
    The original batch is passed through the model and we assert that the outputs are not the same. This is to be used
    in conjunction with an algorithm that augments the data during AFTER_DATALOADER event.

    Assumes gradient accumulation 1.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def after_forward(self, state, logger):
        if state.grad_accum != 1:
            raise ValueError(f'This check assumes grad_accum of 1, got {state.grad_accum}')
        batch_idx = state.timer.batch_in_epoch.value
        batch_size = state.batch_num_samples
        original_batch = self.dataset[batch_idx:batch_idx + batch_size]
        original_outputs = state.model(original_batch)

        assert not torch.allclose(original_outputs[0], state.outputs[0])


class TestTrainerEvents():

    @pytest.fixture
    def config(self):
        return {
            'model': SimpleConvModel(),
            'train_dataloader': DataLoader(
                dataset=RandomImageDataset(size=16),
                batch_size=4,
            ),
            'eval_dataloader': None,
            'max_duration': '1ep',
            'loggers': []
        }

    def test_data_augmented(self, config):
        config['algorithms'] = [CutOut(n_holes=1, length=5)]

        # we give the callback access to the dataset to test
        # that the images have been augmented.
        config['callbacks'] = [
            AssertDataAugmented(dataset=config['train_dataloader'].dataset),
        ]
        trainer = Trainer(**config)
        trainer.fit()

    def test_data_not_augmented(self, config):
        config['callbacks'] = [
            AssertDataAugmented(dataset=config['train_dataloader'].dataset),
        ]
        trainer = Trainer(**config)
        with pytest.raises(AssertionError):
            trainer.fit()


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
            'model': SimpleConvModel(),
            'train_dataloader': DataLoader(
                dataset=RandomImageDataset(size=16),
                batch_size=4,
            ),
            'eval_dataloader': DataLoader(
                dataset=RandomImageDataset(size=16),
                batch_size=4,
            ),
            'max_duration': '2ep',
            'loggers': [],  # no progress bar
        }

    # Note: Not all algorithms, callbacks, and loggers are compatible
    #       with the above configuration. The fixtures below filter and
    #       create the objects to test.

    @pytest.fixture(params=algorithms_registry.items(), ids=algorithms_registry.keys())
    def algorithm(self, request):

        name, hparams = request.param
        skip_list = {
            'swa': 'SWA not compatible with composed schedulers.',
            'alibi': 'Not compatible with conv model.',
            'seq_length_warmup': 'Not compatible with conv model.',
            'randaugment': 'Requires PIL dataset to test.',
            'augmix': 'Required PIL dataset to test.',
            'stochastic_depth': 'Only applies to ResNets.',
            'no_op_model': 'Not compatible with this model.'
        }

        # skip any tests incompatible with this config
        if name in skip_list:
            pytest.skip(skip_list[name])
        elif name in ('cutmix, mixup, label_smoothing'):
            # see: https://github.com/mosaicml/composer/issues/362
            pytest.importorskip("torch", minversion="1.10", reason="Pytorch 1.10 required.")

        # create the algorithms
        if name in ('cutmix, mixup'):  # these algos have required algorithms
            algorithm = hparams(num_classes=2).initialize_object()
        else:
            algorithm = hparams().initialize_object()

        return algorithm

    @pytest.fixture(params=callback_registry.items(), ids=callback_registry.keys())
    def callback(self, request, tmpdir, monkeypatch):
        name, hparams = request.param

        # create callback
        if name == 'run_directory_uploader':
            monkeypatch.setenv("KEY_ENVIRON", str(tmpdir))

            callback = hparams(
                provider='local',
                container='.',
                key_environ="KEY_ENVIRON",
            ).initialize_object()
        else:
            callback = hparams().initialize_object()

        return callback

    @pytest.fixture(params=logger_registry.items(), ids=logger_registry.keys())
    def logger(self, request):

        name, hparams = request.param

        required_args = {}
        if name == 'wandb':
            pytest.importorskip('wandb', reason='Required wandb')

        return hparams(**required_args).initialize_object()

    """
    Tests that training completes.
    """

    def test_algorithms(self, config, algorithm):
        config['algorithms'] = [algorithm]
        trainer = Trainer(**config)
        trainer.fit()

    @pytest.mark.timeout(10)
    def test_callbacks(self, config, callback):
        config['callbacks'] = [callback]
        trainer = Trainer(**config)
        trainer.fit()

    def test_loggers(self, config, logger):
        config['loggers'] = [logger]
        trainer = Trainer(**config)
        trainer.fit()

    """
    Tests that training with multiple fits complete.
    Note: future functional tests should test for
    idempotency (e.g functionally)
    """

    def test_algorithms_multiple_calls(self, config, algorithm):
        if isinstance(algorithm, LayerFreezing):
            pytest.xfail("Known idempotency issue.")
        config['algorithms'] = [algorithm]
        trainer = Trainer(**config)
        self._test_multiple_fits(trainer)

    def test_callbacks_multiple_calls(self, config, callback):
        if isinstance(callback, RunDirectoryUploader):
            pytest.xfail("Known idempotency issue.")
        config['callbacks'] = [callback]
        trainer = Trainer(**config)
        self._test_multiple_fits(trainer)

    def test_loggers_multiple_calls(self, config, logger):
        if isinstance(logger, (FileLogger, WandBLogger)):
            pytest.xfail("Cannot close/load multiple times yet.")
        config['loggers'] = [logger]
        trainer = Trainer(**config)
        self._test_multiple_fits(trainer)

    def _test_multiple_fits(self, trainer):
        trainer.fit()
        trainer.state.max_duration *= 2
        trainer.fit()
