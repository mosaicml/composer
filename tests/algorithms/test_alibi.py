# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from operator import attrgetter

import pytest
import torch

from composer.algorithms.alibi import Alibi, apply_alibi
from composer.core.event import Event
from composer.loggers import Logger
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker


def make_synthetic_state(family):
    synthetic_config = make_dataset_configs(model_family=[family])[0]
    return synthetic_hf_state_maker(synthetic_config)


def _double_batch_sequence_length(batch):
    for k, v in batch.items():
        if v.ndim >= 2:
            batch[k] = torch.cat([v, v], dim=1)
    return batch


def check_number_of_modules_replaced(family, model, replaced_pairs):
    if family == 'gpt2':
        attention_modules = model.config.num_hidden_layers
        # Should "replace" the (1) `GPT2Model` and (n) `GPT2Attention` instances
        expected_pairs = 1 + attention_modules
    elif family == 'bert':
        attention_modules = model.config.num_hidden_layers
        # Should "replace" the (1) `BertEmbeddings` and (n) `BertSelfAttention` instances
        expected_pairs = 1 + attention_modules
    else:
        raise NotImplementedError('Tests not implemented for synthetic_state_family=' + family)
    assert len(replaced_pairs) == expected_pairs


def check_position_embeddings(family, model, max_sequence_length):
    if family == 'gpt2':
        position_embedding_attribute = 'model.transformer.wpe'
    elif family == 'bert':
        position_embedding_attribute = 'model.bert.embeddings.position_embeddings'
    else:
        raise NotImplementedError('Tests not implemented for synthetic_state_family=' + family)

    pos_embedding_module = attrgetter(position_embedding_attribute)(model)
    pos_embedding_weight = getattr(pos_embedding_module, 'weight')

    assert pos_embedding_weight.shape[0] == max_sequence_length
    assert not pos_embedding_weight.requires_grad
    assert torch.max(torch.abs(pos_embedding_weight)) == 0.0


def check_forward_backward(model, batch):
    model.zero_grad()
    output = model.forward(batch)
    output['loss'].backward()


def check_batch_reshaping(before, after, length):
    # Make sure all the batch tensors have the same shape
    input_ids_after_shape = after['input_ids'].shape

    # Just make sure the lengths are correct
    for k in before.keys():

        assert k in after, 'No keys should be removed during sequence reshaping.'

        assert after[
            k].shape == input_ids_after_shape, 'All tensors should have the same size after sequence reshaping.'

        b_numel = before[k].shape[0] * before[k].shape[1]
        a_numel = after[k].shape[0] * after[k].shape[1]
        assert a_numel >= b_numel - length, 'Sequence reshaping should throw away at most curr_sequence_length tokens.'

        import torch
        assert torch.all(after[k][0] == before[k][
            0, :input_ids_after_shape[1]]), 'Sequence reshaping should not change the token order.'

    for k in after.keys():
        assert k in before, 'No keys should be added during sequence reshaping.'


@pytest.mark.parametrize('synthetic_state_family', ['bert', 'gpt2'])
@pytest.mark.parametrize('double_sequence_length', [False, True])
class TestAlibi:

    def test_functional(self, synthetic_state_family: str, double_sequence_length: bool):
        state, _, dataloader = make_synthetic_state(synthetic_state_family)
        if synthetic_state_family == 'gpt2':
            max_sequence_length = state.model.config.n_positions
        elif synthetic_state_family == 'bert':
            max_sequence_length = state.model.config.max_position_embeddings
        else:
            raise NotImplementedError('Tests not implemented for synthetic_state_family=' + synthetic_state_family)

        if double_sequence_length:
            max_sequence_length = 2 * max_sequence_length

        # Apply ALiBi using the functional
        replaced_pairs = apply_alibi(
            model=state.model,
            max_sequence_length=max_sequence_length,
            output_replaced_pairs=True,
        )

        # Ensure that the expected number of modules were affected
        check_number_of_modules_replaced(synthetic_state_family, state.model, replaced_pairs)

        # Ensure that the position embeddings are properly shaped and zeroed
        check_position_embeddings(synthetic_state_family, state.model, max_sequence_length)

        # Try a forward/backward at the max sequence length
        batch = next(iter(dataloader))
        if double_sequence_length:
            batch = _double_batch_sequence_length(batch)
        assert batch['input_ids'].shape[1] == max_sequence_length

        check_forward_backward(state.model, batch)

    @pytest.mark.parametrize('train_sequence_length_scaling', [0.25, 1.0])
    def test_algorithm(self, synthetic_state_family: str, empty_logger: Logger, double_sequence_length: bool,
                       train_sequence_length_scaling: float):
        state, _, dataloader = make_synthetic_state(synthetic_state_family)

        if synthetic_state_family == 'gpt2':
            max_sequence_length = state.model.config.n_positions
        elif synthetic_state_family == 'bert':
            max_sequence_length = state.model.config.max_position_embeddings
        else:
            raise NotImplementedError('Tests not implemented for synthetic_state_family=' + synthetic_state_family)

        if double_sequence_length:
            max_sequence_length = 2 * max_sequence_length

        # Synthetic dataset has a size of 2 batches per epoch (max duration = 1ep)
        alibi = Alibi(
            max_sequence_length=max_sequence_length,
            train_sequence_length_scaling=train_sequence_length_scaling,
        )
        # Apply ALiBi to the model
        alibi.apply(Event.INIT, state, empty_logger)

        batch_before = next(iter(dataloader))
        if double_sequence_length:
            batch_before = _double_batch_sequence_length(batch_before)
        state.batch = deepcopy(batch_before)

        # Apply any batch reshaping
        alibi.apply(Event.AFTER_DATALOADER, state, empty_logger)

        # Ensure proper batch reshaping
        check_batch_reshaping(batch_before, state.batch, int(train_sequence_length_scaling * max_sequence_length))

        # Ensure that the model runs forwards/backwards
        check_forward_backward(state.model, state.batch)
