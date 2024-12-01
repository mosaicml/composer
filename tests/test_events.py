# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from composer import Trainer
from composer.core import Event, Time
from composer.core.time import TimeUnit
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.events import EventCounterCallback


@pytest.fixture(scope='session')
def train_dataset():
    return RandomClassificationDataset(size=16)


@pytest.fixture(scope='session')
def eval_dataset():
    return RandomClassificationDataset(size=16)


@pytest.fixture(scope='session')
def model():
    return SimpleModel()


@pytest.fixture(scope='session')
def optimizer(model):
    return torch.optim.Adam(model.parameters())


@pytest.fixture(scope='session')
def evaluator1(eval_dataset):
    return DataLoader(
        dataset=eval_dataset,
        batch_size=8,
        sampler=dist.get_sampler(eval_dataset),
        num_workers=0,
        drop_last=True,
    )


@pytest.fixture(scope='session')
def evaluator2(eval_dataset):
    return DataLoader(
        dataset=eval_dataset,
        batch_size=4,
        sampler=dist.get_sampler(eval_dataset),
        num_workers=0,
        drop_last=True,
    )


@pytest.fixture
def event_counter_callback():
    return EventCounterCallback()


@pytest.mark.parametrize('event', list(Event))
def test_event_values(event: Event):
    assert event.name.lower() == event.value


@pytest.mark.parametrize(
    'world_size',
    [
        pytest.param(1),
        pytest.param(2, marks=pytest.mark.world_size(2)),
    ],
)
@pytest.mark.parametrize(
    'device,deepspeed_zero_stage,use_fsdp,precision',
    [
        pytest.param('cpu', None, False, 'fp32', id='cpu-ddp'),
        # TODO: Remove filterwarnings after FSDP remove deprecated code
        pytest.param(
            'gpu',
            True,
            False,
            'fp32',
            id='gpu-ddp',
            marks=[
                pytest.mark.gpu,
                pytest.mark.filterwarnings('ignore::UserWarning'),
            ],
        ),
        pytest.param(
            'gpu',
            None,
            True,
            'amp_fp16',
            id='gpu-fsdp',
            marks=[
                pytest.mark.gpu,
                pytest.mark.filterwarnings('ignore::UserWarning'),
            ],
        ),
    ],
)
@pytest.mark.parametrize('save_interval', ['1ep', '1ba'])
def test_event_calls(
    world_size,
    device,
    deepspeed_zero_stage,
    use_fsdp,
    precision,
    save_interval,
    train_dataset,
    eval_dataset,
    model,
    optimizer,
    evaluator1,
    evaluator2,
    event_counter_callback,
):
    with patch.object(Trainer, 'save_checkpoint', return_value=None):
        # mock forward method
        with patch.object(model, 'forward', return_value=torch.tensor(0.0)):
            # initialize the Trainer with the current parameters
            deepspeed_config = None
            if deepspeed_zero_stage:
                deepspeed_config = {'zero_optimization': {'stage': deepspeed_zero_stage}}

            parallelism_config = None
            if use_fsdp:
                parallelism_config = {
                    'fsdp': {
                        'sharding_strategy': 'FULL_SHARD',
                        'mixed_precision': 'PURE',
                        'backward_prefetch': 'BACKWARD_PRE',
                    },
                }

            trainer_instance = Trainer(
                model=model,
                train_dataloader=DataLoader(
                    dataset=train_dataset,
                    batch_size=4,
                    sampler=dist.get_sampler(train_dataset),
                    num_workers=0,
                ),
                eval_dataloader=(evaluator1, evaluator2),
                device_train_microbatch_size=2,
                precision=precision,
                train_subset_num_batches=1,
                eval_subset_num_batches=1,
                max_duration='1ep',
                save_interval=save_interval,
                optimizers=optimizer,
                callbacks=[event_counter_callback],
                device=device,
                deepspeed_config=deepspeed_config,
                parallelism_config=parallelism_config,
            )

            trainer_instance.fit()

            # Assertions
            state = trainer_instance.state

            assert state.dataloader_len is not None
            total_steps = 1 * int(state.dataloader_len)
            batch_size = state.train_dataloader.batch_size  # type: ignore
            assert batch_size is not None
            assert state.device_train_microbatch_size is not None
            total_microbatches = total_steps * math.ceil(batch_size / state.device_train_microbatch_size)

            eval_interval = Time.from_timestring(save_interval)
            if eval_interval.unit == TimeUnit.BATCH:
                total_evals = total_steps // int(eval_interval)
            elif eval_interval.unit == TimeUnit.EPOCH:
                total_evals = 1 // int(eval_interval)
            else:
                total_evals = 0

            if trainer_instance.state.evaluators:
                steps_per_eval = 1
                total_evals_start = total_evals * len(trainer_instance.state.evaluators)
                total_eval_steps = total_evals * steps_per_eval * len(trainer_instance.state.evaluators)
            else:
                total_eval_steps = 0
                total_evals_start = 0

            expected_num_calls = {
                Event.INIT: 1,
                Event.BEFORE_LOAD: 1,
                Event.AFTER_LOAD: 1,
                Event.ITERATION_START: 1,
                Event.EPOCH_START: 1,
                Event.BATCH_START: total_steps,
                Event.BEFORE_DATALOADER: total_steps + 1,  # extra call per epoch when dataloader is exhausted
                Event.AFTER_DATALOADER: total_steps,
                Event.BEFORE_FORWARD: total_microbatches,
                Event.AFTER_FORWARD: total_microbatches,
                Event.BEFORE_LOSS: total_microbatches,
                Event.AFTER_LOSS: total_microbatches,
                Event.BEFORE_BACKWARD: total_microbatches,
                Event.AFTER_BACKWARD: total_microbatches,
                Event.BEFORE_TRAIN_BATCH: total_steps,
                Event.AFTER_TRAIN_BATCH: total_steps,
                Event.BATCH_END: total_steps,
                Event.BATCH_CHECKPOINT: total_steps,
                Event.EPOCH_END: 1,
                Event.EPOCH_CHECKPOINT: 1,
                Event.ITERATION_END: 0,
                Event.ITERATION_CHECKPOINT: 0,
                Event.EVAL_BEFORE_ALL: total_evals,
                Event.EVAL_START: total_evals_start,
                Event.EVAL_BATCH_START: total_eval_steps,
                Event.EVAL_BEFORE_FORWARD: total_eval_steps,
                Event.EVAL_AFTER_FORWARD: total_eval_steps,
                Event.EVAL_BATCH_END: total_eval_steps,
                Event.EVAL_END: total_evals_start,
                Event.EVAL_AFTER_ALL: total_evals,
            }

            for event, expected in expected_num_calls.items():
                actual = event_counter_callback.event_to_num_calls.get(event, 0)
                assert expected == actual, f'{event} call mismatch: {expected} != {actual}'
