# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
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

    eval_subset_num_batches = 5
    train_subset_num_batches = 5

    def get_trainer(self, **kwargs):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        train_dataset = RandomClassificationDataset()
        eval_dataset = RandomClassificationDataset()

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                batch_size=4,
                sampler=dist.get_sampler(train_dataset),
            ),
            eval_dataloader=DataLoader(
                dataset=eval_dataset,
                batch_size=8,
                sampler=dist.get_sampler(eval_dataset),
            ),
            grad_accum=2,
            precision='fp32',
            train_subset_num_batches=self.train_subset_num_batches,
            eval_subset_num_batches=self.eval_subset_num_batches,
            max_duration='2ep',
            optimizers=optimizer,
            callbacks=[EventCounterCallback()],
            **kwargs,
        )

    @pytest.mark.parametrize('world_size', [
        pytest.param(1),
        pytest.param(2, marks=pytest.mark.world_size(2)),
    ])
    @pytest.mark.parametrize('device,deepspeed_zero_stage,use_fsdp', [
        pytest.param('cpu', None, False, id='cpu-ddp'),
        pytest.param('gpu', True, False, id='gpu-ddp', marks=pytest.mark.gpu),
        pytest.param('gpu',
                     None,
                     True,
                     id='gpu-fsdp',
                     marks=[
                         pytest.mark.gpu,
                         pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.12.0'),
                                            reason='requires PyTorch 1.12 or higher')
                     ]),
    ])
    @pytest.mark.parametrize('save_interval', ['1ep', '1ba'])
    def test_event_calls(self, world_size, device, deepspeed_zero_stage, use_fsdp, save_interval):
        save_interval = Time.from_timestring(save_interval)

        deepspeed_config = None
        if deepspeed_zero_stage:
            deepspeed_config = {'zero_optimization': {'stage': deepspeed_zero_stage}}

        fsdp_config = None
        if use_fsdp:
            fsdp_config = {
                'sharding_strategy': 'FULL_SHARD',
                'min_params': 1e8,
                'cpu_offload': False,
                'mixed_precision': 'DEFAULT',
                'backward_prefetch': 'BACKWARD_PRE',
                'activation_checkpointing': False,
                'activation_ocpu_offload': False,
                'verbose': False
            }

        trainer = self.get_trainer(
            device=device,
            deepspeed_config=deepspeed_config,
            fsdp_config=fsdp_config,
            save_interval=save_interval,
            eval_interval=save_interval,
        )
        trainer.fit()

        self._assert_expected_event_calls(trainer, save_interval, num_epochs=2)

    def _assert_expected_event_calls(self, trainer: Trainer, eval_interval: Time, num_epochs: int):
        state = trainer.state

        assert state.dataloader_len is not None
        total_steps = num_epochs * int(state.dataloader_len)
        total_microbatches = total_steps * state.grad_accum

        if eval_interval.unit == TimeUnit.BATCH:
            total_evals = total_steps // int(eval_interval)
        elif eval_interval.unit == TimeUnit.EPOCH:
            total_evals = num_epochs // int(eval_interval)
        else:
            total_evals = 0

        if trainer.state.evaluators:
            steps_per_eval = self.eval_subset_num_batches
            total_eval_steps = total_evals * steps_per_eval * len(trainer.state.evaluators)
        else:
            total_eval_steps = 0

        expected_num_calls = {
            Event.INIT: 1,
            Event.EPOCH_START: num_epochs,
            Event.BATCH_START: total_steps,
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
            Event.EVAL_START: total_evals,
            Event.EVAL_BATCH_START: total_eval_steps,
            Event.EVAL_BEFORE_FORWARD: total_eval_steps,
            Event.EVAL_AFTER_FORWARD: total_eval_steps,
            Event.EVAL_BATCH_END: total_eval_steps,
            Event.EVAL_END: total_evals,
        }

        counter_callback = (cb for cb in trainer.state.callbacks if isinstance(cb, EventCounterCallback))
        counter_callback = next(counter_callback)
        for event, expected in expected_num_calls.items():
            actual = counter_callback.event_to_num_calls[event]
            assert expected == actual, f'{event} call mismatch: {expected} != {actual}'
