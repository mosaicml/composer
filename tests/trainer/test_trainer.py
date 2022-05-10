# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import collections.abc
import contextlib
import copy
import os
import pathlib
from typing import Dict, List, Optional, Union

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from composer import Trainer
from composer.algorithms import CutOut, LabelSmoothing, algorithm_registry
from composer.callbacks import LRMonitor
from composer.core.callback import Callback
from composer.core.evaluator import Evaluator
from composer.core.event import Event
from composer.core.precision import Precision
from composer.core.time import Time, TimeUnit
from composer.datasets import DataLoaderHparams, ImagenetDatasetHparams
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.loggers import FileLogger, WandBLogger
from composer.loggers.in_memory_logger import InMemoryLogger
from composer.models.base import ComposerModel
from composer.optim.scheduler import ExponentialScheduler
from composer.trainer.devices import Device
from composer.trainer.trainer_hparams import callback_registry, logger_registry
from composer.utils import dist
from composer.utils.object_store import ObjectStoreHparams
from tests.algorithms.algorithm_settings import get_settings
from tests.common import (RandomClassificationDataset, RandomImageDataset, SimpleConvModel, SimpleModel, device,
                          world_size)
from tests.common.events import EventCounterCallback
from tests.test_state import assert_state_equivalent


class TestTrainerInit():

    @pytest.fixture
    def model(self):
        return SimpleModel()

    def test_minimal_init(self, model: ComposerModel):
        Trainer(model=model)

    @world_size(1, 2)
    def test_model_ddp_wrapped(self, model: ComposerModel, world_size: int):
        trainer = Trainer(model=model)
        should_be_ddp_wrapped = dist.get_world_size() > 1
        assert isinstance(trainer.state.model, DistributedDataParallel) == should_be_ddp_wrapped

    def test_loggers_before_callbacks(self, model: ComposerModel):
        trainer = Trainer(
            model=model,
            loggers=[InMemoryLogger()],
            callbacks=[LRMonitor()],
            progress_bar=False,
            log_to_console=False,
        )
        assert isinstance(trainer.state.callbacks[0], InMemoryLogger)
        assert isinstance(trainer.state.callbacks[2], LRMonitor)

    def test_invalid_device(self, model: ComposerModel):
        with pytest.raises(ValueError, match="magic_device"):
            Trainer(model=model, device="magic_device")

    @device('gpu', 'cpu')
    def test_optimizer_params_on_device(
        self,
        model: ComposerModel,
        device: str,
    ):
        # Train a model
        train_dataloader = DataLoader(RandomClassificationDataset())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        max_duration = "2ba"
        trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            optimizers=optimizer,
        )
        trainer.fit()

        # Assert that the parameters are on the correct devices
        parameters = trainer.state.optimizers[0].param_groups[0]["params"]
        target_device = 'cuda' if device == 'gpu' else 'cpu'
        assert all(param.device.type == target_device for param in parameters)


class TestTrainerInitOrFit:
    """Validate that certain parameters can be passed in on `Trainer.__init__()` or `Trainer.fit()`"""

    @pytest.fixture
    def train_dataloader(self):
        return DataLoader(dataset=RandomClassificationDataset(), batch_size=2)

    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def max_duration(self):
        return Time(1, TimeUnit.EPOCH)

    @pytest.mark.parametrize("train_subset_num_batches", [-1, 1])
    @pytest.mark.parametrize("compute_training_metrics", [True, False])
    def test_train_dataloader(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
        train_subset_num_batches: int,
        compute_training_metrics: bool,
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the train_dataloader params on Trainer.__init__()
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            train_subset_num_batches=train_subset_num_batches,
            compute_training_metrics=compute_training_metrics,
        )
        init_trainer.fit()

        # Train again with the train_dataloader params specified on Trainer.fit()
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
        )
        fit_trainer.fit(
            train_dataloader=train_dataloader,
            train_subset_num_batches=train_subset_num_batches,
            compute_training_metrics=compute_training_metrics,
        )

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.parametrize("max_duration", [1, "1ep", "1ba", Time(1, TimeUnit.EPOCH)])
    def test_max_duration(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the max_duration param on Trainer.__init__()
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )
        init_trainer.fit()

        # Train again with the max_duration param specified on Trainer.fit()
        fit_trainer = Trainer(
            model=copied_model,
            train_dataloader=train_dataloader,
        )
        fit_trainer.fit(duration=max_duration)

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.parametrize("reset_time", [True, False])
    @pytest.mark.parametrize("new_duration", [
        Time.from_timestring("1ep"),
        Time.from_timestring("1ba"),
        Time.from_timestring("2ep"),
        None,
    ])
    def test_reset_time(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
        new_duration: Time,
        reset_time: bool,
    ):
        # Train once
        trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )
        trainer.fit()

        # Get the timestamp
        first_timestamp = trainer.state.timestamp

        # It should error if the time is not being reset. Otherwise, it should be reset and train OK.
        error_msg = "Please provide the `duration` or specify `reset_time=True`"
        ctx = pytest.raises(ValueError,
                            match=error_msg) if not new_duration and not reset_time else contextlib.nullcontext()
        with ctx:
            # Train again for the same amount of time
            trainer.fit(
                duration=new_duration,
                train_dataloader=train_dataloader,
                reset_time=reset_time,
            )

        # If the fit did not error (new_duration is specified), then assert that the time
        # matches what is expected
        if new_duration is not None:
            if reset_time:
                assert trainer.state.timestamp.get(new_duration.unit) == new_duration
            else:
                first_timestamp_in_new_unit = getattr(first_timestamp, new_duration.unit.name.lower())
                assert trainer.state.timestamp.get(new_duration.unit) == first_timestamp_in_new_unit + new_duration

    @pytest.mark.parametrize("scale_schedule_ratio", [1.0, 2.0])
    @pytest.mark.parametrize("step_schedulers_every_batch", [None, True, False])
    def test_schedulers(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
        scale_schedule_ratio: float,
        step_schedulers_every_batch: Optional[bool],
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the scheduler params on Trainer.__init__()
        scheduler = ExponentialScheduler(2.0)
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            schedulers=scheduler,
            scale_schedule_ratio=scale_schedule_ratio,
            step_schedulers_every_batch=step_schedulers_every_batch,
        )
        init_trainer.fit()

        # Train again with the scheduler params specified on Trainer.fit()
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )
        fit_trainer.fit(
            schedulers=scheduler,
            scale_schedule_ratio=scale_schedule_ratio,
            step_schedulers_every_batch=step_schedulers_every_batch,
        )

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.parametrize("eval_subset_num_batches", [-1, 1])
    @pytest.mark.parametrize("eval_interval", ["1ep", "1ba"])
    @pytest.mark.parametrize(
        "eval_dataloader",
        [
            DataLoader(RandomClassificationDataset(size=2)),  # a normal dataloader
            Evaluator(label='eval', dataloader=DataLoader(RandomClassificationDataset(size=2)),
                      metrics=Accuracy()),  # an evaluator
            [  # multiple evaluators
                Evaluator(label='eval1', dataloader=DataLoader(RandomClassificationDataset(size=2)),
                          metrics=Accuracy()),
                Evaluator(label='eval2', dataloader=DataLoader(RandomClassificationDataset(size=2)), metrics=Accuracy())
            ],
        ])
    def test_eval_dataloader(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
        eval_subset_num_batches: int,
        eval_interval: str,
        eval_dataloader: Union[Evaluator, DataLoader, List[Evaluator]],
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the eval_dataloader params on Trainer.__init__()
        init_event_counter_callback = EventCounterCallback()  # track the number of times eval is called
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            callbacks=[init_event_counter_callback],
            eval_subset_num_batches=eval_subset_num_batches,
            eval_interval=eval_interval,
            progress_bar=False,
            log_to_console=False,
        )
        init_trainer.fit()

        # Train again with the eval_dataloader params specified on Trainer.fit()
        fit_event_counter_callback = EventCounterCallback()  # track the number of times eval is called
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            callbacks=[fit_event_counter_callback],
            progress_bar=False,
            log_to_console=False,
        )
        fit_trainer.fit(
            eval_dataloader=eval_dataloader,
            eval_subset_num_batches=eval_subset_num_batches,
            eval_interval=eval_interval,
        )

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    def test_grad_accum(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
    ):
        grad_accum = 2

        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the grad_accum param on Trainer.__init__()
        init_event_counter_callback = EventCounterCallback()  # track the number of times microbatches are trained
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            grad_accum=grad_accum,
            callbacks=[init_event_counter_callback],
        )
        init_trainer.fit()

        # Train again with the grad_accum param specified on Trainer.fit()
        fit_event_counter_callback = EventCounterCallback()  # track the number of times microbatches are trained
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            callbacks=[fit_event_counter_callback],
        )
        fit_trainer.fit(grad_accum=grad_accum)

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.gpu
    @pytest.mark.parametrize("precision", list(Precision))
    def test_deepspeed(
        self,
        model: ComposerModel,
        precision: Precision,
        max_duration: Time[int],
        train_dataloader: DataLoader,
    ):
        if precision == Precision.BF16:
            pytest.importorskip("torch", minversion="1.10", reason="BF16 precision requires PyTorch 1.10+")

        trainer = Trainer(
            model=model,
            precision=precision,
            deepspeed_config={},
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )

        assert trainer.state.is_model_deepspeed

        trainer.fit()

    @pytest.mark.parametrize("precision", list(Precision))
    @pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
    def test_precision(
        self,
        model: ComposerModel,
        precision: Precision,
        device: str,
        train_dataloader: DataLoader,
        max_duration: Time[int],
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        if precision == Precision.BF16:
            pytest.importorskip("torch", minversion="1.10", reason="BF16 precision requires PyTorch 1.10+")

        should_error = False
        ctx = contextlib.nullcontext()
        if device == "cpu" and precision != Precision.FP32:
            ctx = pytest.raises(ValueError, match="not supproted for CPU training")
            should_error = True
        elif precision == Precision.FP16:
            ctx = pytest.raises(ValueError, match="FP16 precision is only supported when training with DeepSpeed")
            should_error = True

        with ctx:
            # Train once with the precision param on Trainer.__init__()
            init_trainer = Trainer(
                model=model,
                max_duration=max_duration,
                train_dataloader=train_dataloader,
                precision=precision,
            )

        if not should_error:

            init_trainer.fit()

        # Train again with the precision param specified on Trainer.fit()
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )
        with ctx:
            fit_trainer.fit(precision=precision)

        # Assert that the states are equivalent, if we did train
        if not should_error:
            assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.parametrize("grad_clip_norm", [-1.0, 1.0])
    def test_grad_clip_norm(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
        grad_clip_norm: float,
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the grad_clip_norm param on Trainer.__init__()
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            grad_clip_norm=grad_clip_norm,
        )
        init_trainer.fit()

        # Train again with the grad_clip_norm param specified on Trainer.fit()
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )
        fit_trainer.fit(grad_clip_norm=grad_clip_norm)

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.timeout(5.0)
    def test_dataloader_active_iterator_error(self, model: ComposerModel):
        dataloader = DataLoader(
            dataset=RandomClassificationDataset(),
            persistent_workers=True,
            num_workers=1,
        )

        # spin one sample
        _ = next(dataloader.__iter__())

        # Assert the error is raised if the dataloader is specified in init
        with pytest.raises(ValueError, match="active iterator"):
            Trainer(
                model=model,
                train_dataloader=dataloader,
            )

        # Or if the dataloader is specified on fit
        with pytest.raises(ValueError, match="active iterator"):
            trainer = Trainer(model=model)
            trainer.fit(train_dataloader=dataloader)

    def test_multiple_calls_to_fit(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
    ):
        """Test that the trainer supports multiple calls to fit."""
        # Note that callbacks are tested seperately in tests/callbacks/test_callbacks.py
        # To ensure that they support multiple calls of Event.INIT and Event.FIT
        trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )

        # Train once
        trainer.fit()

        # Train again.
        trainer.fit(duration=max_duration)

        assert trainer.state.timestamp.get(max_duration.unit) == 2 * max_duration

    @pytest.mark.parametrize("unit", [TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.SAMPLE])
    def test_training_duration_unit(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        unit: TimeUnit,
    ):
        """Test that the time is correctly set, and events fire correctly, with multiple calls to fit,
        regardless of the time unit"""

        # Construct the trainer
        event_counter_callback = EventCounterCallback()
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            callbacks=[event_counter_callback],
        )

        # Get the batch size
        batch_size = train_dataloader.batch_size
        assert batch_size is not None

        # Get the dataloader length
        dataloader_len = trainer.state.dataloader_len
        assert dataloader_len is not None
        dataloader_len = int(dataloader_len)

        # Get the dataset size
        assert train_dataloader.dataset is not None
        assert isinstance(train_dataloader.dataset, collections.abc.Sized)
        num_samples_per_epoch = len(train_dataloader.dataset)
        assert num_samples_per_epoch % batch_size == 0, "This test assumes no drop_last"

        # Determine the duration (given the unit) and the number of calls to .fit()
        # to train 1 epoch
        if unit == TimeUnit.SAMPLE:
            duration = Time.from_sample(batch_size)
            num_steps_per_epoch = num_samples_per_epoch // batch_size
        elif unit == TimeUnit.BATCH:
            duration = Time.from_batch(1)
            num_steps_per_epoch = dataloader_len
        elif unit == TimeUnit.EPOCH:
            duration = Time.from_epoch(1)
            num_steps_per_epoch = 1
        else:
            raise ValueError(f"Unsupported unit: {unit}")

        # Train for one epoch, incrementally in steps of size `duration`
        for i in range(num_steps_per_epoch):
            # Train for `duration`
            trainer.fit(duration=duration)

            # Determine the number of batches trained
            if unit in (TimeUnit.SAMPLE, TimeUnit.BATCH):
                num_batches_trained = i + 1
            else:
                num_batches_trained = dataloader_len

            # Validate the time
            assert trainer.state.timestamp.batch == num_batches_trained
            assert trainer.state.timestamp.sample == num_batches_trained * batch_size
            assert trainer.state.timestamp.token == 0  # tokens not tracked
            assert trainer.state.timestamp.token_in_epoch == 0  # tokens not tracked

            # Validate the event counter callback
            assert event_counter_callback.event_to_num_calls[Event.EPOCH_START] == 1
            assert event_counter_callback.event_to_num_calls[Event.BATCH_START] == num_batches_trained
            assert event_counter_callback.event_to_num_calls[Event.BATCH_END] == num_batches_trained
            assert event_counter_callback.event_to_num_calls[Event.BATCH_CHECKPOINT] == num_batches_trained

            if num_batches_trained < num_steps_per_epoch:
                # Not yet finished the epoch
                assert trainer.state.timestamp.epoch == 0
                assert trainer.state.timestamp.batch_in_epoch == num_batches_trained
                assert trainer.state.timestamp.sample_in_epoch == num_batches_trained * batch_size
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_END] == 0
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_CHECKPOINT] == 0
            else:
                # Finished the epoch
                assert trainer.state.timestamp.epoch == 1
                assert trainer.state.timestamp.batch_in_epoch == 0
                assert trainer.state.timestamp.sample_in_epoch == 0
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_END] == 1
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_CHECKPOINT] == 1

        # Train for a second epoch
        # Validate that batch_in_epoch / sample_in_epoch are reset properly
        for i in range(num_steps_per_epoch):
            # Train for `duration`
            trainer.fit(duration=duration)

            # Determine the number of batches trained in the epoch
            if unit in (TimeUnit.SAMPLE, TimeUnit.BATCH):
                num_batches_trained = i + 1
            else:
                num_batches_trained = dataloader_len

            # Validate the time
            assert trainer.state.timestamp.batch == dataloader_len + num_batches_trained
            assert trainer.state.timestamp.sample == num_samples_per_epoch + num_batches_trained * batch_size
            assert trainer.state.timestamp.token == 0  # tokens not tracked
            assert trainer.state.timestamp.token_in_epoch == 0  # tokens not tracked

            # Validate the event counter callback
            assert event_counter_callback.event_to_num_calls[Event.EPOCH_START] == 2
            assert event_counter_callback.event_to_num_calls[Event.BATCH_START] == dataloader_len + num_batches_trained
            assert event_counter_callback.event_to_num_calls[Event.BATCH_END] == dataloader_len + num_batches_trained
            assert event_counter_callback.event_to_num_calls[
                Event.BATCH_CHECKPOINT] == dataloader_len + num_batches_trained

            if num_batches_trained < num_steps_per_epoch:
                # Not yet finished the epoch
                assert trainer.state.timestamp.epoch == 1
                assert trainer.state.timestamp.batch_in_epoch == num_batches_trained
                assert trainer.state.timestamp.sample_in_epoch == num_batches_trained * batch_size
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_END] == 1
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_CHECKPOINT] == 1
            else:
                # Finished the epoch
                assert trainer.state.timestamp.epoch == 2
                assert trainer.state.timestamp.batch_in_epoch == 0
                assert trainer.state.timestamp.sample_in_epoch == 0
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_END] == 2
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_CHECKPOINT] == 2


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
        config = copy.deepcopy(config)  # ensure the reference model is not passed to tests

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
        assert trainer.state.timestamp.epoch == "1ep"  # ensure checkpoint state loaded
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
        batch_idx = state.timestamp.batch_in_epoch.value
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
    """The below is a catch-all test that runs the Trainer with each algorithm, callback, and loggers. Success is
    defined as a successful training run.

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
        name, hparams = request.param

        if name == 'mlperf':
            pytest.skip('mlperf callback tested separately.')

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
    def test_algorithm_trains(self, name: str, rank_zero_seed: int, device: str):
        if name in ('no_op_model', 'scale_schedule'):
            pytest.skip('stub algorithms')

        if name in ('cutmix, mixup, label_smoothing'):
            # see: https://github.com/mosaicml/composer/issues/362
            pytest.importorskip("torch", minversion="1.10", reason="Pytorch 1.10 required.")

        setting = get_settings(name)
        if setting is None:
            pytest.xfail('No setting provided in algorithm_settings.')

        trainer = Trainer(
            model=setting['model'],
            train_dataloader=DataLoader(dataset=setting['dataset'], batch_size=4),
            max_duration='2ep',
            loggers=[],
            seed=rank_zero_seed,
            device=device,
        )
        trainer.fit()

        # fit again for another epoch
        trainer.fit(duration='1ep')


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
        except ImportError as e:
            raise ImportError(("Composer was installed without ffcv support. "
                               "To use ffcv with Composer, please install ffcv in your environment.")) from e
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
