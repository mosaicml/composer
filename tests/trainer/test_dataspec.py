# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Any, Optional

import pytest
import torch
from torch.utils.data import DataLoader

from composer import DataSpec, Trainer
from tests.common import RandomClassificationDataset, RandomTextLMDataset, SimpleModel

N = 128


class TestDefaultGetNumSamples:

    @pytest.fixture
    def dataspec(self):
        dataloader = DataLoader(RandomClassificationDataset())
        return DataSpec(dataloader=dataloader)

    # yapf: disable
    @pytest.mark.parametrize(
        'batch', [
            {'a': torch.rand(N, 8), 'b': torch.rand(N, 64)},  # dict
            [{'a': torch.rand(N, 8)}, {'c': torch.rand(N, 64)}],  # list of dict
            (torch.rand(N, 8), torch.rand(N, 64)),  # tuple
            [torch.rand(N, 8), torch.rand(N, 64)],  # list
            torch.rand(N, 8),  # tensor
            torch.rand(N, 8, 4, 2),  # 4-dim tensor
        ],
    )
    # yapf: enable
    def test_num_samples_infer(self, batch: Any, dataspec: DataSpec):
        assert dataspec._default_get_num_samples_in_batch(batch) == N

    def test_batch_dict_mismatch(self, dataspec: DataSpec):
        N = 128
        batch = {'a': torch.rand(N, 8), 'b': torch.rand(N * 2, 64)}

        with pytest.raises(NotImplementedError, match='multiple Tensors'):
            dataspec._default_get_num_samples_in_batch(batch)

    def test_unable_to_infer(self, dataspec: DataSpec):
        N = 128
        batch = [torch.rand(N, 8), 'I am a string.']

        with pytest.raises(ValueError, match='Unable to determine'):
            dataspec._default_get_num_samples_in_batch(batch)


@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('sequence_length', [8])
@pytest.mark.parametrize('use_keys', [True, False])
@pytest.mark.parametrize('set_attr', [True, False])
def test_get_num_tokens_hf_default(batch_size: int, sequence_length: int, use_keys: bool, set_attr: bool):
    dataset = RandomTextLMDataset(size=20, sequence_length=sequence_length, use_keys=use_keys)
    if set_attr:
        dataset.max_seq_len = sequence_length  # type: ignore
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataspec = DataSpec(dataloader=dataloader)

    batch = next(iter(dataloader))
    actual = dataspec._default_get_num_tokens_in_batch(batch)

    if not use_keys and not set_attr:
        expected = 0
    else:
        expected = sequence_length * batch_size
    assert actual == expected


@pytest.mark.parametrize(
    'return_dict,requested_key,expected',
    [
        [True, None, 8],  # dict with default key
        [False, None, 8],  # int with default key
        [False, 'loss_generating', 8],  # int with non-default key
        [True, 'loss_generating', 4],  # dict with non-default key
    ],
)
def test_get_num_tokens_types(return_dict: bool, requested_key: Optional[str], expected: Optional[int]):
    should_error = expected is None
    error_context = pytest.raises(ValueError) if should_error else contextlib.nullcontext()

    def get_num_tokens_in_batch(batch):
        num_tokens = 8
        num_loss_generating_tokens = 4

        if return_dict:
            return {'total': num_tokens, 'loss_generating': num_loss_generating_tokens}

        return num_tokens

    dataspec = DataSpec(dataloader=[], get_num_tokens_in_batch=get_num_tokens_in_batch)

    batch = {}
    extra_args = {}
    if requested_key is not None:
        extra_args['token_type'] = requested_key

    with error_context:
        actual = dataspec.get_num_tokens_in_batch(batch, **extra_args)
        assert actual == expected


def test_small_batch_at_end_warning():
    batch_size = 4
    dataset_size = 17
    eval_batch_size = 2
    eval_dataset_size = 25

    model = SimpleModel()
    trainer = Trainer(
        model=model,
        eval_interval=f'2ba',
        train_dataloader=DataLoader(RandomClassificationDataset(size=dataset_size), batch_size=batch_size),
        eval_dataloader=DataLoader(RandomClassificationDataset(size=eval_dataset_size), batch_size=eval_batch_size),
        max_duration=f'8ba',
    )

    with pytest.warns(UserWarning, match='Cannot split tensor of length.*'):
        trainer.fit()


@pytest.mark.parametrize(
    'batch,num_samples',
    [
        [
            {
                'a': torch.rand(N, 8),
                'b': torch.rand(N, 64),
            },
            N,
        ],  # dict
        [
            [
                {
                    'a': torch.rand(N, 8),
                },
                {
                    'c': torch.rand(N, 64),
                },
            ],
            N,
        ],  # list of dict
        [
            {
                'a': [1, 2],
                'b': [3, 4],
            },
            2,
        ],  # dict of lists
        [(torch.rand(N, 8), torch.rand(N, 64)), N],  # tuple
        [[torch.rand(N, 8), torch.rand(N, 64)], N],  # list
        [torch.rand(N, 8), N],  # tensor
        [torch.rand(N, 8, 4, 2), N],  # 4-dim tensor
    ],
)
def test_num_samples_in_batch(batch, num_samples):
    data_spec = DataSpec(dataloader=DataLoader(RandomClassificationDataset(size=17), batch_size=4))
    assert data_spec.get_num_samples_in_batch(batch) == num_samples
