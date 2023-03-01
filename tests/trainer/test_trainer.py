# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import collections.abc
import contextlib
import copy
import datetime
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Union

import pytest
import torch
from packaging import version
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from composer import Callback, Evaluator, Trainer
from composer.algorithms import CutOut, LabelSmoothing
from composer.core import Event, Precision, State, Time, TimeUnit
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.imagenet import build_ffcv_imagenet_dataloader
from composer.devices import Device
from composer.loggers import InMemoryLogger, Logger, RemoteUploaderDownloader
from composer.loss import soft_cross_entropy
from composer.models import ComposerModel
from composer.optim import ExponentialScheduler
from composer.trainer.trainer import _generate_run_name
from composer.utils import dist, is_model_deepspeed, is_model_fsdp, map_collection, reproducibility
from tests.common import (InfiniteClassificationDataset, RandomClassificationDataset, RandomImageDataset,
                          SimpleConvModel, SimpleModel, device, world_size)
from tests.common.events import EventCounterCallback
from tests.test_state import assert_state_equivalent


class SleepyCallback(Callback):

    def __init__(self, sleep_duration: datetime.timedelta, event: Event) -> None:
        self.sleep_duration = sleep_duration
        self.event = event

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == self.event:
            time.sleep(self.sleep_duration.total_seconds())


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

    def test_invalid_device(self, model: ComposerModel):
        with pytest.raises(ValueError, match='magic_device'):
            Trainer(model=model, device='magic_device')

    @world_size(1, 2)
    @device('gpu', 'cpu')
    def test_gpu_logging(self, model: ComposerModel, world_size: int, device: str):
        in_mem_logger = InMemoryLogger()
        Trainer(model=model, loggers=[in_mem_logger])
        expected_key, expected_value = f'num_{device}s_per_node', world_size
        assert expected_key in in_mem_logger.hyperparameters
        assert in_mem_logger.hyperparameters[expected_key] == expected_value

    @device('gpu', 'cpu')
    def test_optimizer_params_on_device(
        self,
        model: ComposerModel,
        device: str,
    ):
        # Train a model
        train_dataset = RandomClassificationDataset()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        max_duration = '2ba'
        trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=DataLoader(train_dataset, sampler=dist.get_sampler(train_dataset)),
            optimizers=optimizer,
        )
        trainer.fit()

        # Assert that the parameters are on the correct devices
        parameters = trainer.state.optimizers[0].param_groups[0]['params']
        target_device = 'cuda' if device == 'gpu' else 'cpu'
        assert all(param.device.type == target_device for param in parameters)


def _assert_optimizer_is_on_device(optimizer: torch.optim.Optimizer):
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                assert v.device.type == 'cuda'


def _get_classification_dataloader():
    dataset = RandomClassificationDataset(size=2)
    return DataLoader(dataset, sampler=dist.get_sampler(dataset))


class TestTrainerInitOrFit:
    """Validate that certain parameters can be passed in on `Trainer.__init__()` or `Trainer.fit()`"""

    @pytest.fixture
    def train_dataloader(self):
        dataset = RandomClassificationDataset(size=10)
        return DataLoader(dataset=dataset, batch_size=2, sampler=dist.get_sampler(dataset))

    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def max_duration(self):
        return Time(1, TimeUnit.EPOCH)

    @pytest.mark.parametrize('train_subset_num_batches', [-1, 1])
    def test_train_dataloader(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
        train_subset_num_batches: int,
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the train_dataloader params on Trainer.__init__()
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            train_subset_num_batches=train_subset_num_batches,
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
        )

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.parametrize('max_duration', [1, '1ep', '1ba', Time(1, TimeUnit.EPOCH)])
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

    @pytest.mark.parametrize('max_duration', [1, '1ep', '1ba', '1sp'])
    @pytest.mark.parametrize('train_subset_num_batches', [-1, 1])
    def test_infinite_train_loader(self, model: ComposerModel, max_duration: Union[int, str],
                                   train_subset_num_batches: int):
        should_raise = (isinstance(max_duration, int) or
                        max_duration.endswith('ep')) and (train_subset_num_batches is None or
                                                          train_subset_num_batches == -1)
        context = pytest.raises(
            ValueError,
            match='max_duration cannot be specified in epochs') if should_raise else contextlib.nullcontext()
        with context:
            train_loader = DataLoader(InfiniteClassificationDataset(), batch_size=4)
            trainer = Trainer(model=model,
                              train_dataloader=train_loader,
                              max_duration=max_duration,
                              train_subset_num_batches=train_subset_num_batches)
            trainer.fit()

    @pytest.mark.parametrize('reset_time', [True, False])
    @pytest.mark.parametrize('new_duration', [
        Time.from_timestring('1ep'),
        Time.from_timestring('1ba'),
        Time.from_timestring('2ep'),
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
        error_msg = 'Please provide the `duration` or specify `reset_time=True`'
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

    @pytest.mark.parametrize('scale_schedule_ratio', [1.0, 2.0])
    @pytest.mark.parametrize('step_schedulers_every_batch', [None, True, False])
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

    @pytest.mark.parametrize('eval_subset_num_batches', [-1, 1])
    @pytest.mark.parametrize('eval_interval', ['1ep', '1ba'])
    @pytest.mark.parametrize(
        'eval_dataloader',
        [
            _get_classification_dataloader(),  # a normal dataloader
            Evaluator(
                label='eval',
                dataloader=_get_classification_dataloader(),
                metric_names=['Accuracy'],
            ),  # an evaluator
            [  # multiple evaluators
                Evaluator(
                    label='eval1',
                    dataloader=_get_classification_dataloader(),
                    metric_names=['Accuracy'],
                ),
                Evaluator(
                    label='eval2',
                    dataloader=_get_classification_dataloader(),
                    metric_names=['Accuracy'],
                ),
            ],
        ],
    )
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
        )
        init_trainer.fit()

        # Train again with the eval_dataloader params specified on Trainer.fit()
        fit_event_counter_callback = EventCounterCallback()  # track the number of times eval is called
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            callbacks=[fit_event_counter_callback],
        )
        fit_trainer.fit(
            eval_dataloader=eval_dataloader,
            eval_subset_num_batches=eval_subset_num_batches,
            eval_interval=eval_interval,
        )

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    def test_microbatch(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
    ):
        device_train_microbatch_size = 1

        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the device_train_microbatch_size param on Trainer.__init__()
        init_event_counter_callback = EventCounterCallback()  # track the number of times microbatches are trained
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            device_train_microbatch_size=device_train_microbatch_size,
            callbacks=[init_event_counter_callback],
        )
        init_trainer.fit()

        # Train again with the device_train_microbatch_size param specified on Trainer.fit()
        fit_event_counter_callback = EventCounterCallback()  # track the number of times microbatches are trained
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            callbacks=[fit_event_counter_callback],
        )
        fit_trainer.fit(device_train_microbatch_size=device_train_microbatch_size)

        # Assert that the states are equivalent
        assert_state_equivalent(init_trainer.state, fit_trainer.state)

    @pytest.mark.gpu
    @pytest.mark.filterwarnings(
        "ignore:Setting `device_train_microbatch_size='auto'` is an experimental feature which may cause uncaught Cuda Out of Memory errors. In this case, please manually set device_train_microbatch_size explicitly to an integer instead."
    )
    @pytest.mark.parametrize('dataloader_in_init', [True, False])
    def test_auto_microbatch(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        max_duration: Time[int],
        dataloader_in_init: bool,
    ):
        # Copy the model so the fit_trainer can start with the same parameter values as the init_trainer
        copied_model = copy.deepcopy(model)

        # Train once with the device_train_microbatch_size=1
        init_event_counter_callback = EventCounterCallback()  # track the number of times microbatches are trained
        init_trainer = Trainer(
            model=model,
            max_duration=max_duration,
            train_dataloader=train_dataloader if dataloader_in_init else None,
            device_train_microbatch_size='auto',
            callbacks=[init_event_counter_callback],
        )
        init_trainer.fit(train_dataloader=train_dataloader if not dataloader_in_init else None)

        # Train again with the device_train_microbatch_size='auto'
        fit_event_counter_callback = EventCounterCallback()  # track the number of times microbatches are trained
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader if dataloader_in_init else None,
            callbacks=[fit_event_counter_callback],
        )
        fit_trainer.fit(
            train_dataloader=train_dataloader if not dataloader_in_init else None,
            device_train_microbatch_size='auto',
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
    @pytest.mark.parametrize('precision', [Precision.FP32, Precision.AMP_BF16, Precision.AMP_FP16])
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_deepspeed(
        self,
        model: ComposerModel,
        precision: Precision,
        max_duration: Time[int],
        train_dataloader: DataLoader,
    ):
        trainer = Trainer(
            model=model,
            precision=precision,
            deepspeed_config={},
            max_duration=max_duration,
            train_dataloader=train_dataloader,
        )

        assert is_model_deepspeed(trainer.state.model)

        assert trainer.state.deepspeed_enabled
        trainer.fit()

    @pytest.mark.gpu
    @pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                        reason='requires PyTorch 1.13 or higher')
    @pytest.mark.parametrize('precision', [Precision.FP32, Precision.AMP_BF16, Precision.AMP_FP16])
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_fsdp(
        self,
        model: ComposerModel,
        precision: Precision,
        max_duration: Time[int],
        train_dataloader: DataLoader,
    ):
        if precision == Precision.FP32:  # FSDP FULL_SHARD doesn't support FP32
            return

        fsdp_config = {
            'sharding_strategy': 'FULL_SHARD',
            'min_params': 1e8,
            'cpu_offload': False,
            'mixed_precision': 'PURE',
            'backward_prefetch': 'BACKWARD_PRE',
            'activation_checkpointing': False,
            'activation_cpu_offload': False,
            'verbose': False
        }

        # Need to catch the case where we try to train
        # with precision FP16.
        ctx = contextlib.nullcontext()
        should_error = False

        with ctx:
            trainer = Trainer(
                model=model,
                precision=precision,
                fsdp_config=fsdp_config,
                max_duration=max_duration,
                train_dataloader=train_dataloader,
            )

        if not should_error:
            assert is_model_fsdp(trainer.state.model)

            assert trainer.state.fsdp_enabled
            trainer.fit()

    @pytest.mark.gpu
    def test_device(
        self,
        model: ComposerModel,
        max_duration: Time[int],
        train_dataloader: DataLoader,
    ):
        trainer = Trainer(model=model, device='gpu', max_duration=max_duration, train_dataloader=train_dataloader)
        # Run fit to ensure there are no device mismatches
        trainer.fit()

        # Finally assert the devices are correct
        assert all(p.device.type == 'cuda' for p in trainer.state.model.parameters())
        map_collection(trainer.state.optimizers, _assert_optimizer_is_on_device)

    @pytest.mark.gpu
    def test_device_with_checkpoint(
        self,
        model: ComposerModel,
        tmp_path: pathlib.Path,
        max_duration: Time[int],
        train_dataloader: DataLoader,
    ):
        copied_model = copy.deepcopy(model)
        trainer = Trainer(model=model, device='gpu', max_duration=max_duration, train_dataloader=train_dataloader)
        checkpoint_path = str(tmp_path / 'checkpoint.pt')
        trainer.save_checkpoint(checkpoint_path)

        trainer_2 = Trainer(model=copied_model,
                            load_path=checkpoint_path,
                            max_duration=max_duration,
                            train_dataloader=train_dataloader)
        # Run fit to ensure there are no device mismatches
        trainer_2.fit(reset_time=True)

        # And ensure the device on the new trainer is correct
        assert all(p.device.type == 'cuda' for p in trainer_2.state.model.parameters())
        map_collection(trainer_2.state.optimizers, _assert_optimizer_is_on_device)

    @pytest.mark.parametrize('precision', [Precision.FP32, Precision.AMP_BF16, Precision.AMP_FP16])
    @pytest.mark.parametrize('device', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
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

        should_error = False
        ctx = contextlib.nullcontext()
        if device == 'cpu' and precision != Precision.FP32:
            ctx = pytest.raises(ValueError, match='not supported for CPU training.')
            should_error = True

        with ctx:
            # Train once with the precision param on Trainer.__init__()
            init_trainer = Trainer(
                model=model,
                max_duration=max_duration,
                train_dataloader=train_dataloader,
                precision=precision,
                device=device,
            )

        if not should_error:

            init_trainer.fit()

        # Train again with the precision param specified on Trainer.fit()
        fit_trainer = Trainer(
            model=copied_model,
            max_duration=max_duration,
            train_dataloader=train_dataloader,
            device=device,
        )
        with ctx:
            fit_trainer.fit(precision=precision)

        # Assert that the states are equivalent, if we did train
        if not should_error:
            assert_state_equivalent(init_trainer.state, fit_trainer.state)

    def test_dataloader_active_iterator_error(self, model: ComposerModel):
        dataset = RandomClassificationDataset()
        dataloader = DataLoader(
            dataset=dataset,
            persistent_workers=True,
            num_workers=1,
            sampler=dist.get_sampler(dataset),
        )

        # spin one sample
        _ = next(dataloader.__iter__())

        # Assert the error is raised if the dataloader is specified in init
        with pytest.raises(ValueError, match='active iterator'):
            Trainer(
                model=model,
                train_dataloader=dataloader,
            )

        # Or if the dataloader is specified on fit
        with pytest.raises(ValueError, match='active iterator'):
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

    @pytest.mark.parametrize('eval_interval', ['1ba', '1ep'])
    def test_eval_is_excluded_from_wct_tracking(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
        eval_interval: str,
    ):
        # Construct the trainer with a callback that sleeps during evaluation
        sleep_duration = datetime.timedelta(seconds=0.05)
        sleepy_callback = SleepyCallback(
            sleep_duration=sleep_duration,
            event=Event.EVAL_AFTER_FORWARD,
        )
        event_counter_callback = EventCounterCallback()
        dataset = RandomClassificationDataset()
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            train_subset_num_batches=2,  # make training fast
            eval_dataloader=DataLoader(
                dataset=dataset,
                batch_size=2,
                sampler=dist.get_sampler(dataset),
            ),
            callbacks=[sleepy_callback, event_counter_callback],
            eval_interval=eval_interval,
            max_duration='2ep',
            eval_subset_num_batches=1,
        )

        # Train
        trainer.fit()

        # Validate that eval was called
        expected_num_evals = 4 if eval_interval == '1ba' else 2
        assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == expected_num_evals

        # Validate the timestamps.
        # Training duration should be less than the sleeping
        assert trainer.state.timestamp.total_wct < sleep_duration * expected_num_evals
        # The last evaluation duration should be at least as much as the sleeping
        assert trainer.state.eval_timestamp.total_wct > sleep_duration

    @pytest.mark.world_size(2)
    def test_wct_consistency_across_ranks(
        self,
        train_dataloader: DataLoader,
        model: ComposerModel,
    ):
        """Test that the wct is the same across multiple ranks"""
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration='1ba',
        )

        trainer.fit()

        # First check that the timestamp is non-zero
        timestamp = trainer.state.timestamp
        assert timestamp.total_wct.total_seconds() > 0
        assert timestamp.epoch_wct.total_seconds() > 0
        assert timestamp.batch_wct.total_seconds() > 0

        # Validate it is the same across ranks
        my_timestamp_tensor = torch.tensor([
            timestamp.total_wct.total_seconds(),
            timestamp.epoch_wct.total_seconds(),
            timestamp.batch_wct.total_seconds(),
        ],
                                           dtype=torch.float64)
        rank_zero_timestamp_tensor = torch.tensor([
            timestamp.total_wct.total_seconds(),
            timestamp.epoch_wct.total_seconds(),
            timestamp.batch_wct.total_seconds(),
        ],
                                                  dtype=torch.float64)
        dist.broadcast(rank_zero_timestamp_tensor, src=0)
        assert torch.all(my_timestamp_tensor == rank_zero_timestamp_tensor)

    @pytest.mark.parametrize('unit', [TimeUnit.EPOCH, TimeUnit.BATCH, TimeUnit.SAMPLE])
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
        assert num_samples_per_epoch % batch_size == 0, 'This test assumes no drop_last'

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
            raise ValueError(f'Unsupported unit: {unit}')

        current_epoch_time = datetime.timedelta(seconds=0)

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
            assert trainer.state.timestamp.total_wct > current_epoch_time

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
                assert trainer.state.timestamp.epoch_wct > current_epoch_time
                assert trainer.state.timestamp.epoch_wct == trainer.state.timestamp.total_wct
                if i > 0:
                    assert trainer.state.timestamp.epoch_wct > trainer.state.timestamp.batch_wct
                else:
                    assert trainer.state.timestamp.epoch_wct == trainer.state.timestamp.batch_wct
            else:
                # Finished the epoch
                assert trainer.state.timestamp.epoch == 1
                assert trainer.state.timestamp.batch_in_epoch == 0
                assert trainer.state.timestamp.sample_in_epoch == 0
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_END] == 1
                assert event_counter_callback.event_to_num_calls[Event.EPOCH_CHECKPOINT] == 1
                assert trainer.state.timestamp.epoch_wct == datetime.timedelta(seconds=0)
                assert trainer.state.timestamp.batch_wct == datetime.timedelta(seconds=0)

            current_epoch_time = trainer.state.timestamp.total_wct

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
            assert trainer.state.timestamp.total_wct > trainer.state.timestamp.batch_wct
            assert trainer.state.timestamp.total_wct > trainer.state.timestamp.epoch_wct

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


@pytest.mark.vision
class TestFFCVDataloaders:

    train_file = None
    val_file = None
    tmp_path = None

    @pytest.fixture(autouse=True)
    def create_dataset(self, tmp_path_factory: pytest.TempPathFactory):
        dataset_train = RandomImageDataset(size=16, is_PIL=True)
        self.tmp_path = tmp_path_factory.mktemp('ffcv')
        output_train_file = str(self.tmp_path / 'train.ffcv')
        write_ffcv_dataset(dataset_train, write_path=output_train_file, num_workers=1, write_mode='proportion')
        dataset_val = RandomImageDataset(size=16, is_PIL=True)
        output_val_file = str(self.tmp_path / 'val.ffcv')
        write_ffcv_dataset(dataset_val, write_path=output_val_file, num_workers=1, write_mode='proportion')
        self.train_file = output_train_file
        self.val_file = output_val_file

    def _get_dataloader(self, is_train):
        assert self.tmp_path is not None
        assert self.train_file is not None
        assert self.val_file is not None
        datadir = os.path.join(self.tmp_path, self.train_file if is_train else self.val_file)
        return build_ffcv_imagenet_dataloader(
            datadir=str(datadir),
            global_batch_size=4,
            is_train=is_train,
            num_workers=0,
        )

    @pytest.fixture
    def config(self):
        try:
            import ffcv
        except ImportError as e:
            raise ImportError(('Composer was installed without ffcv support. '
                               'To use ffcv with Composer, please install ffcv in your environment.')) from e
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


@world_size(1, 2)
@device('cpu', 'gpu', 'gpu-amp', precision=True)
class TestTrainerEquivalence():

    default_threshold = {'atol': 0, 'rtol': 0}
    reference_model = None
    reference_folder = None

    def assert_models_equal(self, model_1, model_2, threshold=None):
        if threshold is None:
            threshold = self.default_threshold

        assert model_1 is not model_2, 'Same model should not be compared.'
        for param1, param2 in zip(model_1.parameters(), model_2.parameters()):
            torch.testing.assert_close(param1, param2, **threshold)

    @pytest.fixture
    def config(self, device: Device, precision: Precision, world_size: int, rank_zero_seed: int):
        """Returns the reference config."""

        train_dataset = RandomClassificationDataset(size=16)
        eval_dataset = RandomClassificationDataset(size=16)

        return {
            'model':
                SimpleModel(),
            'train_dataloader':
                DataLoader(
                    dataset=train_dataset,
                    batch_size=4,
                    sampler=dist.get_sampler(train_dataset),
                ),
            'eval_dataloader':
                DataLoader(
                    dataset=eval_dataset,
                    sampler=dist.get_sampler(eval_dataset),
                ),
            'max_duration':
                '2ep',
            'seed':
                rank_zero_seed,
            'device':
                device,
            'precision':
                precision,
            'loggers': [],  # no progress bar
        }

    @pytest.fixture(autouse=True)
    def create_reference_model(self, config, tmp_path_factory: pytest.TempPathFactory, *args):
        """Trains the reference model, and saves checkpoints."""
        config = copy.deepcopy(config)  # ensure the reference model is not passed to tests

        save_folder = tmp_path_factory.mktemp('{device}-{precision}'.format(**config))
        config.update({'save_interval': '1ep', 'save_folder': str(save_folder), 'save_filename': 'ep{epoch}.pt'})

        trainer = Trainer(**config)
        trainer.fit()

        self.reference_model = trainer.state.model
        self.reference_folder = save_folder

    def test_determinism(self, config, *args):
        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_microbatch_size(self, config, precision, *args):
        # microbatching requires non-zero tolerance
        # Precision.AMP requires a even higher tolerance.
        threshold = {
            'atol': 1e-04 if precision == Precision.AMP_FP16 else 1e-05,
            'rtol': 1e-02 if precision == Precision.AMP_FP16 else 1e-04,
        }

        config.update({
            'device_train_microbatch_size': 2,
        })

        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model, threshold=threshold)

    def test_grad_accum_size(self, config, precision, *args):
        # grad accum requires non-zero tolerance
        # Precision.AMP_FP16 requires a even higher tolerance.
        threshold = {
            'atol': 1e-04 if precision == Precision.AMP_FP16 else 1e-05,
            'rtol': 1e-02 if precision == Precision.AMP_FP16 else 1e-04,
        }

        config.update({
            'grad_accum': 2,
        })

        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model, threshold=threshold)

    def test_max_duration(self, config, *args):
        num_batches = 2 * len(config['train_dataloader'])  # convert 2ep to batches
        config['max_duration'] = f'{num_batches}ba'

        trainer = Trainer(**config)
        trainer.fit()
        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_checkpoint(self, config, *args):
        # load from epoch 1 checkpoint and finish training
        assert self.reference_folder is not None
        checkpoint_file = os.path.join(self.reference_folder, 'ep1.pt')
        config['load_path'] = checkpoint_file

        trainer = Trainer(**config)
        assert trainer.state.timestamp.epoch == '1ep'  # ensure checkpoint state loaded
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_tuple_loss_trainer(self, config, *args):

        def tuple_loss(outputs, targets, *args, **kwargs):
            loss1 = 0.25 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            loss2 = 0.75 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            return (loss1, loss2)

        config['model']._loss_fn = tuple_loss

        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_dict_loss_trainer(self, config, *args):

        def dict_loss(outputs, targets, *args, **kwargs):
            losses = {}
            losses['cross_entropy1'] = 0.25 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            losses['cross_entropy2'] = 0.75 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            return losses

        config['model']._loss_fn = dict_loss

        trainer = Trainer(**config)
        trainer.fit()

        self.assert_models_equal(trainer.state.model, self.reference_model)

    def test_dict_loss_total_trainer(self, config, *args):

        def dict_loss_total(outputs, targets, *args, **kwargs):
            losses = {}
            losses['cross_entropy1'] = 2 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            losses['cross_entropy2'] = 3 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            losses['total'] = soft_cross_entropy(outputs, targets, *args, **kwargs)
            return losses

        config['model']._loss_fn = dict_loss_total

        trainer = Trainer(**config)
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

    Assumes only one microbatch is used.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def after_forward(self, state, logger):
        if state.using_device_microbatch_size and state.device_train_microbatch_size != state.train_dataloader.batch_size:  # type: ignore
            raise ValueError('This check assumes device_train_microbatch_size == batch_size')
        elif not state.using_device_microbatch_size and state.grad_accum != 1:
            raise ValueError(f'This check assumes grad_accum of 1, got {state.grad_accum}')
        batch_idx = state.timestamp.batch_in_epoch.value
        batch_size = len(state.batch[0])
        original_batch = self.dataset[batch_idx:batch_idx + batch_size]
        original_outputs = state.model(original_batch)

        assert not torch.allclose(original_outputs[0], state.outputs[0])


class TestTrainerEvents():

    @pytest.fixture
    def config(self, rank_zero_seed: int):
        dataset = RandomImageDataset(size=16)
        return {
            'model': SimpleConvModel(),
            'train_dataloader': DataLoader(
                dataset=dataset,
                batch_size=4,
                sampler=dist.get_sampler(dataset),
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


@pytest.mark.world_size(2)
def test_state_run_name():
    # seeding with the global rank to ensure that each rank has a different seed
    reproducibility.seed_all(dist.get_global_rank())

    run_name = _generate_run_name()
    # The run name should be the same on every rank -- it is set via a distributed reduction
    # Manually verify that all ranks have the same run name
    run_names = dist.all_gather_object(run_name)
    assert len(run_names) == 2  # 2 ranks
    assert all(run_name == run_names[0] for run_name in run_names)


class TestAutoresumeCompatibility:

    def get_logger(self,
                   tmp_path: pathlib.Path,
                   num_concurrent_uploads: int = 1,
                   file_path_format_string: Optional[str] = None):
        """Returns an object store logger that saves locally."""
        remote_dir = str(tmp_path / 'object_store')
        os.makedirs(remote_dir, exist_ok=True)

        return RemoteUploaderDownloader(bucket_uri='libcloud://.',
                                        backend_kwargs={
                                            'provider': 'local',
                                            'container': '.',
                                            'provider_kwargs': {
                                                'key': remote_dir,
                                            },
                                        },
                                        num_concurrent_uploads=num_concurrent_uploads,
                                        use_procs=False,
                                        upload_staging_folder=str(tmp_path / 'staging_folder'),
                                        **({
                                            'file_path_format_string': file_path_format_string
                                        } if file_path_format_string is not None else {}))

    @pytest.fixture
    def config(self):
        """Returns the reference config."""

        train_dataset = RandomClassificationDataset()
        eval_dataset = RandomClassificationDataset()

        return {
            'model':
                SimpleModel(),
            'train_dataloader':
                DataLoader(
                    dataset=train_dataset,
                    batch_size=4,
                    sampler=dist.get_sampler(train_dataset),
                ),
            'eval_dataloader':
                DataLoader(
                    dataset=eval_dataset,
                    sampler=dist.get_sampler(eval_dataset),
                ),
            'max_duration':
                '2ep',
            'autoresume':
                True,
            'loggers': [],
        }

    def test_autoresume_and_concurrent_uploads_error(self, tmp_path: pathlib.Path, config: Dict[str, Any]):
        pytest.importorskip('libcloud')
        config.update({
            'run_name': 'autoresume_concurrent_uploads_run',
            'save_folder': str(tmp_path / 'checkpoints'),
            'loggers': [self.get_logger(tmp_path, num_concurrent_uploads=2),
                        self.get_logger(tmp_path)]
        })

        # Test that trainer errors out if autoresume is set, and RemoteUploaderDownloader does multiple concurrent uploads.
        # The root cause of this is that it is possible for an updated symlink file to be uploaded before the corresponding
        # checkpoint has finished uploading, and then the run dies, leaving the symlink contents pointing to a checkpoint that
        # does not exist
        with pytest.raises(ValueError,
                           match='Multiple concurrent uploads is not currently supported when using autoresume'):
            _ = Trainer(**config)

    def test_latest_and_object_format_string_error(self, tmp_path: pathlib.Path, config: Dict[str, Any]):
        pytest.importorskip('libcloud')
        config.update({
            'run_name':
                'latest_format_string_run',
            'save_folder':
                str(tmp_path / 'checkpoints'),
            'loggers': [
                self.get_logger(tmp_path, file_path_format_string='test/{remote_file_name}'),
                self.get_logger(tmp_path)
            ]
        })

        # Test that trainer errors out if save_latest_filename is set, and RemoteUploaderDownloader file_path_format_string
        # is not default. The root cause of this is that the symlink file contents are created outside of the RemoteUploaderDownloader
        # and do not take into account its path formatting
        with pytest.raises(
                ValueError,
                match='Specifying a `file_path_format_string` to a `RemoteUploaderDownloader` is not currently supported'
        ):
            _ = Trainer(**config)

        # Ensure that if save_latest_filename is not set, it does not error
        config.update({'save_latest_filename': None, 'autoresume': False})
        _ = Trainer(**config)

    def test_autoresume_and_default_remote_uploader_downloader(self, tmp_path: pathlib.Path, config: Dict[str, Any]):
        pytest.importorskip('libcloud')
        config.update({
            'run_name': 'autoresume_default_remote_ud_run',
            'save_folder': str(tmp_path / 'checkpoints'),
            'loggers': [self.get_logger(tmp_path), self.get_logger(tmp_path)]
        })

        # Just test that the default args for everything do not hit the above errors
        _ = Trainer(**config)
