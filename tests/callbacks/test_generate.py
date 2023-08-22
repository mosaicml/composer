# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import Mock

import pytest
import torch
from packaging import version

from composer.callbacks import Generate
from composer.trainer import Trainer
from composer.utils import dist
from tests.common.datasets import dummy_gpt_lm_dataloader
from tests.common.markers import device, world_size
from tests.common.models import configure_tiny_gpt2_hf_model


@device('cpu', 'gpu')
@pytest.mark.parametrize('use_fsdp', [True, False])
@world_size(1, 2)
def test_generate_callback(device, world_size, use_fsdp):
    if use_fsdp and version.parse(torch.__version__) < version.parse('1.13.0'):
        pytest.skip('FSDP requires torch >= 1.13.0')
    if device == 'cpu' and use_fsdp:
        pytest.skip('FSDP is not supported on CPU.')
    if world_size == 1 and use_fsdp:
        pytest.xfail((
            'Generation with world size 1 and FSDP fails with '
            '`RuntimeError: The tensor has a non-zero number of elements, '
            'but its data is not allocated yet. Caffe2 uses a lazy allocation, '
            'so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.` '
            'This issue is resolved with world size > 1 by a dummy call to forward (see HuggingFaceModel.dummy_forward_called), '
            'but for some reason fails with world size 1.'))
    if device == 'cpu' and world_size > 1:
        pytest.xfail(
            'GPT2 is not currently supported with DDP. See https://github.com/huggingface/transformers/issues/22482 for more details.'
        )

    fsdp_config = None
    if use_fsdp:
        fsdp_config = {
            'sharding_strategy': 'FULL_SHARD',
        }

    model = configure_tiny_gpt2_hf_model()
    model.generate = Mock(wraps=model.generate)
    prompts = ['a', 'b', 'c']

    prompt_batch_size = 2
    gen_interval = 2
    generate_cb = Generate(prompts, interval=f'{gen_interval}ba', batch_size=prompt_batch_size, max_length=5)
    generate_cb.generate = Mock(wraps=generate_cb.generate)

    train_batches = 6
    trainer = Trainer(model=model,
                      train_dataloader=dummy_gpt_lm_dataloader(),
                      device=device,
                      max_duration=f'{train_batches}ba',
                      callbacks=generate_cb,
                      fsdp_config=fsdp_config)

    trainer.logger.log_table = Mock()

    trainer.fit()

    expected_cb_call_count = math.ceil(train_batches / gen_interval)

    # Assert that the generate callback has been called the correct number of times.
    assert generate_cb.generate.call_count == expected_cb_call_count

    # Assert that model.generate has been called the correct number of times to ensure that prompt batching is correct.
    assert model.generate.call_count == math.ceil(len(prompts) / prompt_batch_size) * expected_cb_call_count

    # Assert that log_table is called on the 0th rank only.
    if dist.get_global_rank() == 0:
        assert trainer.logger.log_table.call_count == expected_cb_call_count
    else:
        trainer.logger.log_table.assert_not_called()
