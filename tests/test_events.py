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


@pytest.mark.parametrize('event', list(Event))
def test_event_values(event: Event):
    assert event.name.lower() == event.value


class TestEventCalls:

    eval_subset_num_batches = 1
    train_subset_num_batches = 1

    def get_trainer(self, precision='fp32', max_duration='1ep', save_interval='1ep', **kwargs):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        train_dataset = RandomClassificationDataset(size=16)
        eval_dataset = RandomClassificationDataset(size=16)
        train_batch_size = 4

        evaluator1 = DataLoader(
            dataset=eval_dataset,
            batch_size=8,
            sampler=dist.get_sampler(eval_dataset),
            num_workers=0,
            drop_last=True,
        )

        evaluator2 = DataLoader(
            dataset=eval_dataset,
            batch_size=4,
            sampler=dist.get_sampler(eval_dataset),
            num_workers=0,
            drop_last=True,
        )

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=train_dataset,
                batch_size=train_batch_size,
                sampler=dist.get_sampler(train_dataset),
                num_workers=0,
            ),
            eval_dataloader=(evaluator1, evaluator2),
            device_train_microbatch_size=train_batch_size // 2,
            precision=precision,
            train_subset_num_batches=self.train_subset_num_batches,
            eval_subset_num_batches=self.eval_subset_num_batches,
            max_duration=max_duration,
            save_interval=save_interval,
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
        'device,use_fsdp,precision',
        [
            pytest.param('cpu', False, 'fp32', id='cpu-ddp'),
            # TODO: Remove filterwarnings after FSDP remove deprecated code
            pytest.param(
                'gpu',
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
    def test_event_calls(self, world_size, device, use_fsdp, precision, save_interval):
        # handle 1ba save interval separately to optimize speed
        if save_interval == '1ba':
            # mock the save_checkpoint method to speed up batch saves
            with patch('composer.trainer.trainer.Trainer.save_checkpoint') as mock_save:
                mock_save.return_value = None
                self._run_event_calls_test(
                    world_size,
                    device,
                    use_fsdp,
                    precision,
                    save_interval,
                    num_epochs=1,
                )
        else:
            self._run_event_calls_test(
                world_size,
                device,
                use_fsdp,
                precision,
                save_interval,
                num_epochs=1,
            )

    def _run_event_calls_test(
        self,
        world_size,
        device,
        use_fsdp,
        precision,
        save_interval,
        num_epochs,
    ):

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
            parallelism_config=parallelism_config,
            save_interval=save_interval,
            eval_interval=Time.from_timestring(save_interval),
        )
        trainer.fit()

        self._assert_expected_event_calls(trainer, Time.from_timestring(save_interval), num_epochs=num_epochs)

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
