# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.utils.data import DataLoader

from composer import Trainer
from composer.core import Event, Time
from composer.core.time import TimeUnit
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.events import EventCounterCallback


@pytest.mark.parametrize('event', list(Event))
def test_event_values(event: Event):
    assert event.name.lower() == event.value


class TestEventCalls:

    eval_subset_num_batches = 2
    train_subset_num_batches = 2

    def get_trainer(self, precision='fp32', **kwargs):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        train_dataset = RandomClassificationDataset()
        eval_dataset = RandomClassificationDataset()
        train_batch_size = 4

        evaluator1 = DataLoader(
            dataset=eval_dataset,
            batch_size=8,
            sampler=dist.get_sampler(eval_dataset),
        )

        evaluator2 = DataLoader(
            dataset=eval_dataset,
            batch_size=4,
            sampler=dist.get_sampler(eval_dataset),
        )

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                sampler=dist.get_sampler(train_dataset),
            ),
            eval_dataloader=(evaluator1, evaluator2),
            device_train_microbatch_size=train_batch_size // 2,
            precision=precision,
            train_subset_num_batches=self.train_subset_num_batches,
            eval_subset_num_batches=self.eval_subset_num_batches,
            max_duration='2ep',
            optimizers=optimizer,
            callbacks=[EventCounterCallback()],
            **kwargs,
        )

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
    def test_event_calls(self, world_size, device, deepspeed_zero_stage, use_fsdp, precision, save_interval):
        save_interval = Time.from_timestring(save_interval)

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

        trainer = self.get_trainer(
            precision=precision,
            device=device,
            deepspeed_config=deepspeed_config,
            parallelism_config=parallelism_config,
            save_interval=save_interval,
            eval_interval=save_interval,
        )
        trainer.fit()

        self._assert_expected_event_calls(trainer, save_interval, num_epochs=2)

    def _assert_expected_event_calls(self, trainer: Trainer, eval_interval: Time, num_epochs: int):
        state = trainer.state

        assert state.dataloader_len is not None
        total_steps = num_epochs * int(state.dataloader_len)
        batch_size = state.train_dataloader.batch_size  # type: ignore
        assert batch_size is not None
        assert state.device_train_microbatch_size is not None
        total_microbatches = total_steps * math.ceil(batch_size / state.device_train_microbatch_size)

        if eval_interval.unit == TimeUnit.BATCH:
            total_evals = total_steps // int(eval_interval)
        elif eval_interval.unit == TimeUnit.EPOCH:
            total_evals = num_epochs // int(eval_interval)
        else:
            total_evals = 0

        if trainer.state.evaluators:
            steps_per_eval = self.eval_subset_num_batches
            total_evals_start = total_evals * len(trainer.state.evaluators)
            total_eval_steps = total_evals * steps_per_eval * len(trainer.state.evaluators)
        else:
            total_eval_steps = 0
            total_evals_start = 0

        expected_num_calls = {
            Event.INIT: 1,
            Event.BEFORE_LOAD: 1,
            Event.AFTER_LOAD: 1,
            Event.ITERATION_START: 1,
            Event.EPOCH_START: num_epochs,
            Event.BATCH_START: total_steps,
            Event.BEFORE_DATALOADER: total_steps + num_epochs,  # extra call per epoch when dataloader is exhausted
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
            Event.EPOCH_END: num_epochs,
            Event.EPOCH_CHECKPOINT: num_epochs,
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

        counter_callback = (cb for cb in trainer.state.callbacks if isinstance(cb, EventCounterCallback))
        counter_callback = next(counter_callback)
        for event, expected in expected_num_calls.items():
            actual = counter_callback.event_to_num_calls[event]
            assert expected == actual, f'{event} call mismatch: {expected} != {actual}'
