# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from composer.callbacks import Generate
from composer.core import Event
from composer.trainer import Trainer
from composer.utils import dist
from tests.common.datasets import dummy_gpt_lm_dataloader
from tests.common.markers import device, world_size
from tests.common.models import configure_tiny_gpt2_hf_model


@device('cpu', 'gpu')
@pytest.mark.parametrize('use_fsdp', [True, False])
@world_size(1, 2)
class TestGenerate():

    def _check_test_params(self, device, world_size, use_fsdp) -> None:
        if device == 'cpu' and use_fsdp:
            pytest.skip('FSDP is not supported on CPU.')
        if world_size == 1 and use_fsdp:
            pytest.xfail((
                'Generation with world size 1 and FSDP fails with '
                '`RuntimeError: The tensor has a non-zero number of elements, '
                'but its data is not allocated yet. Caffe2 uses a lazy allocation, '
                'so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.` '
                'This issue is resolved with world size > 1 by a dummy call to forward (see HuggingFaceModel.dummy_forward_called), '
                'but for some reason fails with world size 1.'
            ))
        if device == 'cpu' and world_size > 1:
            pytest.xfail(
                'GPT2 is not currently supported with DDP. See https://github.com/huggingface/transformers/issues/22482 for more details.',
            )

    def _create_trainer(self, device, max_duration, use_fsdp, generate_cb: Optional[Generate] = None) -> Trainer:
        return Trainer(
            model=configure_tiny_gpt2_hf_model(),
            train_dataloader=dummy_gpt_lm_dataloader(),
            device=device,
            max_duration=max_duration,
            callbacks=generate_cb,
            parallelism_config={'fsdp': {
                'sharding_strategy': 'FULL_SHARD'
            }} if use_fsdp else None,
        )

    def test_no_effect_on_training(self, device, world_size, use_fsdp):
        torch.manual_seed(0)

        self._check_test_params(device, world_size, use_fsdp)

        max_duration = '6ba'

        trainer_ref = self._create_trainer(device, max_duration, use_fsdp)
        trainer_ref.fit()

        trainer_generate = self._create_trainer(
            device,
            max_duration,
            use_fsdp,
            generate_cb=Generate(prompts=['a', 'bc', 'defg'], interval=f'2ba', batch_size=2, max_new_tokens=5),
        )
        trainer_generate.fit()

        model_ref = trainer_ref.state.model
        model_generate = trainer_generate.state.model

        # Assert that the models should be equivalent
        assert model_ref is not model_generate, 'Same model should not be compared.'
        for param1, param2 in zip(model_ref.parameters(), model_generate.parameters()):
            torch.testing.assert_close(param1, param2)

    def test_calls(self, device, world_size, use_fsdp):
        self._check_test_params(device, world_size, use_fsdp)

        # Create generate callback
        prompts = ['a', 'bc', 'defg']
        prompt_batch_size = 2
        gen_interval = 2
        generate_cb = Generate(prompts, interval=f'{gen_interval}ba', batch_size=prompt_batch_size, max_new_tokens=5)

        # Create trainer
        train_batches = 6
        trainer = self._create_trainer(device, f'{train_batches}ba', use_fsdp, generate_cb)

        # Mock methods
        state = trainer.state
        model = state.model.module if state.is_model_ddp else state.model
        model.generate = Mock(wraps=model.generate)  # type: ignore
        generate_cb.generate = Mock(wraps=generate_cb.generate)
        trainer.logger.log_table = Mock()

        trainer.fit()

        expected_cb_call_count = math.ceil(train_batches / gen_interval)

        # Assert that the generate callback has been called the correct number of times.
        assert generate_cb.generate.call_count == expected_cb_call_count

        # Assert that model.generate has been called the correct number of times to ensure that prompt batching is correct.
        assert model.generate.call_count == math.ceil(  # type: ignore
            len(prompts) / prompt_batch_size,
        ) * expected_cb_call_count

        # Assert that log_table is called on the 0th rank only.
        if dist.get_global_rank() == 0:
            assert trainer.logger.log_table.call_count == expected_cb_call_count
        else:
            trainer.logger.log_table.assert_not_called()

    def test_calls_end_of_training(self, device, world_size, use_fsdp):
        self._check_test_params(device, world_size, use_fsdp)

        prompts = ['a', 'bc', 'defg']
        prompt_batch_size = 2
        gen_interval = 2
        generate_cb = Generate(prompts, interval=f'{gen_interval}ba', batch_size=prompt_batch_size, max_new_tokens=5)

        # Create trainer with gen_interval > max_duration
        train_batches = 1
        trainer = self._create_trainer(device, f'{train_batches}ba', use_fsdp, generate_cb)

        # Mock methods
        state = trainer.state
        model = state.model.module if state.is_model_ddp else state.model
        model.generate = Mock(wraps=model.generate)  # type: ignore
        generate_cb.generate = Mock(wraps=generate_cb.generate)
        trainer.logger.log_table = Mock()

        trainer.fit()

        expected_cb_call_count = 1

        # Assert that the generate callback has been called ONLY once
        assert generate_cb.generate.call_count == expected_cb_call_count

        # An additional fit call should not trigger additional calls to generate
        trainer.engine.run_event(Event.FIT_END)
        assert generate_cb.generate.call_count == expected_cb_call_count
