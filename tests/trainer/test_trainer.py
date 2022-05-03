# Copyright 2021 MosaicML. All Rights Reserved.

import contextlib
import os
import pathlib
from copy import deepcopy
from typing import Dict

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from composer import Trainer
from composer.algorithms import CutOut, LabelSmoothing, algorithm_registry
from composer.callbacks import CheckpointSaver, LRMonitor
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.precision import Precision
from composer.datasets import DataLoaderHparams, ImagenetDatasetHparams
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.loggers import FileLogger, ProgressBarLogger, WandBLogger
from composer.trainer.devices.device import Device
from composer.trainer.trainer_hparams import callback_registry, logger_registry
from composer.utils import MissingConditionalImportError, dist
from composer.utils.object_store import ObjectStoreHparams
from tests.algorithms.algorithm_settings import get_settings
from tests.common import (RandomClassificationDataset, RandomImageDataset, SimpleConvModel, SimpleModel, device,
                          world_size)


class TestTrainerInit():

    @pytest.fixture
    def config(self, rank_zero_seed: int):
        return {
            'model': SimpleModel(),
            'train_dataloader': DataLoader(dataset=RandomClassificationDataset()),
            'eval_dataloader': DataLoader(dataset=RandomClassificationDataset()),
            'max_duration': '2ep',
            'seed': rank_zero_seed,
        }

    @pytest.mark.gpu
    @pytest.mark.parametrize("precision", list(Precision))
    def test_precision(self, config, precision: Precision):
        config['precision'] = precision
        config['device'] = 'gpu'

        if precision == Precision.BF16:
            pytest.importorskip("torch", minversion="1.10", reason="BF16 precision requires PyTorch 1.10+")

        with pytest.raises(ValueError) if precision == Precision.FP16 else contextlib.nullcontext():
            Trainer(**config)

    @pytest.mark.gpu
    @pytest.mark.parametrize("precision", list(Precision))
    def test_trainer_with_deepspeed(self, config, precision: Precision):
        config['deepspeed_config'] = {}
        config['precision'] = precision
        config['device'] = 'gpu'

        if precision == Precision.BF16:
            pytest.importorskip("torch", minversion="1.10", reason="BF16 precision requires PyTorch 1.10+")

        trainer = Trainer(**config)

        assert trainer.deepspeed_enabled

        trainer.fit()

    def test_init(self, config):
        trainer = Trainer(**config)
        assert isinstance(trainer, Trainer)

    def test_model_ddp_wrapped(self, config):
        trainer = Trainer(**config)
        should_be_ddp_wrapped = dist.get_world_size() > 1 and "deepspeed_config" not in config
        assert isinstance(trainer.state.model, DistributedDataParallel) == should_be_ddp_wrapped

    def test_loggers_before_callbacks(self, config):
        config.update({
            "loggers": [ProgressBarLogger()],
            "callbacks": [LRMonitor()],
        })

        trainer = Trainer(**config)
        assert isinstance(trainer.state.callbacks[0], ProgressBarLogger)
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

    @pytest.mark.timeout(5.0)
    def test_init_with_integers(self, config, tmpdir: pathlib.Path):
        config.update({
            'max_duration': 1,
            'save_interval': 10,
            'save_folder': str(tmpdir),
        })

        trainer = Trainer(**config)
        assert trainer.state.max_duration == "1ep"
        checkpoint_saver = None
        for callback in trainer.state.callbacks:
            if isinstance(callback, CheckpointSaver):
                checkpoint_saver = callback
        assert checkpoint_saver is not None
        trainer.state.timer.epoch._value = 10
        assert checkpoint_saver.save_interval(trainer.state, Event.EPOCH_CHECKPOINT)

    @pytest.mark.timeout(5.0)
    def test_init_with_max_duration_in_batches(self, config):
        config["max_duration"] = '1ba'
        trainer = Trainer(**config)
        assert trainer.state.max_duration is not None
        assert trainer.state.max_duration.to_timestring() == "1ba"


@world_size(1, 2)
@device('cpu', 'gpu', 'gpu-amp', precision=True)
@pytest.mark.timeout(15)  # higher timeout as each model is trained twice
class TestTrainerEquivalence():

    reference_model: torch.nn.Module
    reference_folder: pathlib.Path
    default_threshold: Dict[str, float]

    def assert_models_equal(self, model_1, model_2, threshold=None):
        if threshold is None:
            threshold = self.default_threshold

        assert model_1 is not model_2, "Same model should not be compared."
        for param1, param2 in zip(model_1.parameters(), model_2.parameters()):
            torch.testing.assert_allclose(param1, param2, **threshold)

    @pytest.fixture(autouse=True)
    def set_default_threshold(self, device, precision, world_size):
        """Sets the default threshold to 0.

        Individual tests can override by passing thresholds directly to assert_models_equal.
        """
        self.default_threshold = {'atol': 0, 'rtol': 0}

    @pytest.fixture
    def config(self, device: Device, precision: Precision, world_size: int, rank_zero_seed: int):
        """Returns the reference config."""

        return {
            'model': SimpleModel(),
            'train_dataloader': DataLoader(
                dataset=RandomClassificationDataset(),
                batch_size=4,
                shuffle=False,
            ),
            'eval_dataloader': DataLoader(
                dataset=RandomClassificationDataset(),
                shuffle=False,
            ),
            'max_duration': '2ep',
            'seed': rank_zero_seed,
            'device': device,
            'precision': precision,
            'loggers': [],  # no progress bar
        }

    @pytest.fixture(autouse=True)
    def create_reference_model(self, config, tmpdir_factory, *args):
        """Trains the reference model, and saves checkpoints."""
        config = deepcopy(config)  # ensure the reference model is not passed to tests

        save_folder = tmpdir_factory.mktemp("{device}-{precision}".format(**config))
        config.update({'save_interval': '1ep', 'save_folder': str(save_folder), 'save_filename': 'ep{epoch}.pt'})

        trainer = Trainer(**config)
        trainer.fit()

        self.reference_model = trainer.state.model
        self.reference_folder = save_folder

    def test_determinism(self, config, *args):
        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_grad_accum(self, config, precision, *args):
        # grad accum requires non-zero tolerance
        # Precision.AMP requires a even higher tolerance.
        threshold = {
            'atol': 1e-04 if precision == Precision.AMP else 1e-08,
            'rtol': 1e-02 if precision == Precision.AMP else 1e-05,
        }

        config.update({
            'grad_accum': 2,
        })

        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model, threshold=threshold)

    def test_max_duration(self, config, *args):
        num_batches = 2 * len(config["train_dataloader"])  # convert 2ep to batches
        config['max_duration'] = f'{num_batches}ba'

        trainer = Trainer(**config)
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

    def test_algorithm_different(self, config, *args):
        # as a control, we train with an algorithm and
        # expect the test to fail
        config['algorithms'] = [LabelSmoothing(0.1)]
        trainer = Trainer(**config)
        trainer.fit()

        with pytest.raises(AssertionError):
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
    def config(self, rank_zero_seed: int):
        return {
            'model': SimpleConvModel(),
            'train_dataloader': DataLoader(
                dataset=RandomImageDataset(size=16),
                batch_size=4,
            ),
            'eval_dataloader': None,
            'max_duration': '1ep',
            'loggers': [],
            'seed': rank_zero_seed,
        }

    def test_data_augmented(self, config):
        config['algorithms'] = [CutOut()]

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


@pytest.mark.timeout(15)
class TestTrainerAssets:
    """
    The below is a catch-all test that runs the Trainer
    with each algorithm, callback, and loggers. Success
    is defined as a successful training run.

    This should eventually be replaced by functional
    tests for each object, in situ of our trainer.

    We use the hparams_registry associated with our
    config management to retrieve the objects to test.
    """

    @pytest.fixture(params=[1, 2], ids=['ga-1', 'ga-2'])
    def config(self, rank_zero_seed: int, request):
        grad_accum = request.param

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
            'seed': rank_zero_seed,
            'grad_accum': grad_accum,
        }

    # Note: Not all algorithms, callbacks, and loggers are compatible
    #       with the above configuration. The fixtures below filter and
    #       create the objects to test.

    @pytest.fixture(params=callback_registry.items(), ids=tuple(callback_registry.keys()))
    def callback(self, request):
        _, hparams = request.param

        callback = hparams().initialize_object()

        return callback

    @pytest.fixture(params=logger_registry.items(), ids=tuple(logger_registry.keys()))
    def logger(self, request, tmpdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch):

        name, hparams = request.param

        remote_dir = str(tmpdir / "remote_dir")
        os.makedirs(remote_dir)
        local_dir = str(tmpdir / "local_dir")
        os.makedirs(local_dir)
        monkeypatch.setenv("OBJECT_STORE_KEY", remote_dir)  # for the local option, the key is the path
        provider_hparams = ObjectStoreHparams(
            provider='local',
            key_environ="OBJECT_STORE_KEY",
            container=".",
        )

        required_args = {}
        if name == 'wandb':
            pytest.importorskip('wandb', reason='Required wandb')
        if name == 'object_store':
            required_args['object_store_hparams'] = provider_hparams
            required_args['use_procs'] = False

        if name == 'object_store_logger':
            monkeypatch.setenv("KEY_ENVIRON", str(tmpdir))

            logger = hparams(
                provider='local',
                container='.',
                key_environ="KEY_ENVIRON",
            ).initialize_object()
        else:
            logger = hparams(**required_args).initialize_object()

        return logger

    """
    Tests that training completes.
    """

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

    def test_callbacks_multiple_calls(self, config, callback):
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


class TestTrainerAlgorithms:

    @pytest.mark.parametrize("name", algorithm_registry.list_algorithms())
    @pytest.mark.timeout(5)
    @device('gpu')
    def test_algorithm_trains(self, name, rank_zero_seed, device):
        if name in ('no_op_model', 'scale_schedule'):
            pytest.skip('stub algorithms')

        if name in ('cutmix, mixup, label_smoothing'):
            # see: https://github.com/mosaicml/composer/issues/362
            pytest.importorskip("torch", minversion="1.10", reason="Pytorch 1.10 required.")

        setting = get_settings(name)
        if setting is None:
            pytest.skip('No setting provided in algorithm_settings.')

        trainer = Trainer(
            model=setting['model'],
            train_dataloader=DataLoader(dataset=setting['dataset'], batch_size=4),
            max_duration='2ep',
            loggers=[],
            seed=rank_zero_seed,
            device=device,
        )
        trainer.fit()


@pytest.mark.vision
@pytest.mark.timeout(30)
class TestFFCVDataloaders:
    train_file: str
    val_file: str
    tmpdir: str

    @pytest.fixture(autouse=True)
    def create_dataset(self, tmpdir_factory):
        dataset_train = RandomImageDataset(size=16, is_PIL=True)
        output_train_file = str(tmpdir_factory.mktemp("ffcv").join("train.ffcv"))
        write_ffcv_dataset(dataset_train, write_path=output_train_file, num_workers=1, write_mode='proportion')
        dataset_val = RandomImageDataset(size=16, is_PIL=True)
        tmp_dir = tmpdir_factory.mktemp("ffcv")
        output_val_file = str(tmp_dir.join("val.ffcv"))
        write_ffcv_dataset(dataset_val, write_path=output_val_file, num_workers=1, write_mode='proportion')
        self.train_file = output_train_file
        self.val_file = output_val_file
        self.tmpdir = str(tmp_dir)

    def _get_dataloader(self, is_train):
        dl_hparams = DataLoaderHparams(num_workers=0)
        ds_hparams = ImagenetDatasetHparams(is_train=is_train,
                                            use_ffcv=True,
                                            ffcv_dir=self.tmpdir,
                                            ffcv_dest=self.train_file if is_train else self.val_file)
        return ds_hparams.initialize_object(batch_size=4, dataloader_hparams=dl_hparams)

    @pytest.fixture
    def config(self):
        try:
            import ffcv  # type: ignore
        except ImportError:
            raise MissingConditionalImportError(extra_deps_group="ffcv", conda_package="ffcv")
        train_dataloader = self._get_dataloader(is_train=True)
        val_dataloader = self._get_dataloader(is_train=False)
        assert isinstance(train_dataloader, ffcv.Loader)
        assert isinstance(val_dataloader, ffcv.Loader)
        return {
            'model': SimpleConvModel(),
            'train_dataloader': train_dataloader,
            'eval_dataloader': val_dataloader,
            'max_duration': '2ep',
        }

    """
    Tests that training completes with ffcv dataloaders.
    """

    @device('gpu-amp', precision=True)
    def test_ffcv(self, config, device, precision):
        config['device'] = device
        config['precision'] = precision
        trainer = Trainer(**config)
        trainer.fit()
