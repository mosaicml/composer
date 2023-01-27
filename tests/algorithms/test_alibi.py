# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from operator import attrgetter

import pytest
import torch

from composer.algorithms.alibi import Alibi, apply_alibi
from composer.core import Event, State
from composer.devices import DeviceCPU
from composer.loggers import Logger
from tests.common.datasets import dummy_bert_lm_dataloader, dummy_gpt_lm_dataloader
from tests.common.models import configure_tiny_bert_hf_model, configure_tiny_gpt2_hf_model


def _double_batch_sequence_length(batch):
    for k, v in batch.items():
        if v.ndim >= 2:
            batch[k] = torch.cat([v, v], dim=1)
    return batch


def check_position_embeddings(model, max_sequence_length):
    transformers = pytest.importorskip('transformers')
    if isinstance(model.config, transformers.GPT2Config):
        position_embedding_attribute = 'model.transformer.wpe'
    elif isinstance(model.config, transformers.BertConfig):
        position_embedding_attribute = 'model.bert.embeddings.position_embeddings'
    else:
        raise NotImplementedError('Tests not implemented for model with config=' + str(type(model.config)))

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


def encountered_alibi_warning(caplog):
    """Return true if the caplog shows an alibi warning in the log"""
    for (logger_name, level, _) in caplog.record_tuples:
        if 'alibi' in logger_name and level >= 30:  # Warnings are level 30
            return True
    return False


def test_warning_is_triggered(caplog):
    """Test that Alibi triggers a warning when it has no effect."""
    pytest.importorskip('transformers')
    apply_alibi(
        model=torch.nn.Sequential(torch.nn.Linear(20, 10), torch.nn.Linear(10, 5)),
        max_sequence_length=64,
    )
    assert encountered_alibi_warning(caplog), 'A warning should be generated when Alibi has no effect.'


def test_registry(caplog):
    """Test that registry additions are used by Alibi."""
    pytest.importorskip('transformers')
    from composer.algorithms.alibi.attention_surgery_functions import policy_registry

    @policy_registry.register(torch.nn.Linear)
    def zero_linear_weights(  # pyright: reportUnusedFunction = none
            module: torch.nn.Module, idx: int, max_sequence_length: int) -> torch.nn.Module:
        assert isinstance(module, torch.nn.Linear)
        old_weight = getattr(module, 'weight')
        new_weight = torch.nn.Parameter(torch.zeros_like(old_weight))
        setattr(module, 'weight', new_weight)
        return module

    apply_alibi(
        model=torch.nn.Sequential(torch.nn.Linear(20, 10), torch.nn.Linear(10, 5)),
        max_sequence_length=64,
    )
    assert not encountered_alibi_warning(caplog), 'No warnings should be generated after adding to the registry.'
    del (policy_registry[torch.nn.Linear])


@pytest.mark.parametrize('model,dataloader', [(configure_tiny_bert_hf_model, dummy_bert_lm_dataloader),
                                              (configure_tiny_gpt2_hf_model, dummy_gpt_lm_dataloader)])
class TestAlibi:

    def test_functional(
        self,
        model,
        dataloader,
        caplog,
    ):
        transformers = pytest.importorskip('transformers')
        model = model()
        if isinstance(model.config, transformers.GPT2Config):
            max_sequence_length = model.config.n_positions
        elif isinstance(model.config, transformers.BertConfig):
            max_sequence_length = model.config.max_position_embeddings
        else:
            raise NotImplementedError('Tests not implemented for model with config=' + str(type(model.config)))

        dataloader = dataloader(sequence_length=max_sequence_length)

        #### With default sequence length ####

        # Apply ALiBi using the functional
        apply_alibi(
            model=model,
            max_sequence_length=max_sequence_length,
        )
        assert not encountered_alibi_warning(caplog)  # This should not generate any warnings

        # Ensure that the position embeddings are properly shaped and zeroed
        check_position_embeddings(model, max_sequence_length)

        # Try a forward/backward at the max sequence length
        batch = next(iter(dataloader))
        assert batch['input_ids'].shape[1] == max_sequence_length

        check_forward_backward(model, batch)

        #### With double sequence length ####

        # Apply ALiBi using the functional
        apply_alibi(
            model=model,
            max_sequence_length=2 * max_sequence_length,
        )
        assert not encountered_alibi_warning(caplog)  # This should not generate any warnings

        # Ensure that the position embeddings are properly shaped and zeroed
        check_position_embeddings(model, 2 * max_sequence_length)

        # Try a forward/backward at the max sequence length
        batch = next(iter(dataloader))
        batch = _double_batch_sequence_length(batch)
        assert batch['input_ids'].shape[1] == 2 * max_sequence_length

        check_forward_backward(model, batch)

    @pytest.mark.parametrize('train_sequence_length_scaling', [0.25, 1.0])
    def test_algorithm(self, model, dataloader, empty_logger: Logger, train_sequence_length_scaling: float, caplog,
                       request: pytest.FixtureRequest):
        transformers = pytest.importorskip('transformers')

        model = model()
        dataloader = dataloader()
        state = State(
            model=model,
            rank_zero_seed=0,
            run_name='run_name',
            device=DeviceCPU(),
            dataloader=dataloader,
            dataloader_label='train',
            max_duration='1ep',
        )

        if isinstance(model.config, transformers.GPT2Config):
            max_sequence_length: int = state.model.config.n_positions  # type: ignore
        elif isinstance(model.config, transformers.BertConfig):
            max_sequence_length: int = state.model.config.max_position_embeddings  # type: ignore
        else:
            raise NotImplementedError('Tests not implemented for model with config=' + str(type(model.config)))

        # Synthetic dataset has a size of 2 batches per epoch (max duration = 1ep)
        alibi = Alibi(
            max_sequence_length=max_sequence_length,
            train_sequence_length_scaling=train_sequence_length_scaling,
        )
        # Apply ALiBi to the model
        alibi.apply(Event.INIT, state, empty_logger)
        assert not encountered_alibi_warning(caplog)  # This should not generate any warnings

        batch_before = next(iter(dataloader))
        state.batch = deepcopy(batch_before)

        # Apply any batch reshaping
        alibi.apply(Event.AFTER_DATALOADER, state, empty_logger)

        # Ensure proper batch reshaping
        check_batch_reshaping(batch_before, state.batch, int(train_sequence_length_scaling * max_sequence_length))

        # Ensure that the model runs forwards/backwards
        check_forward_backward(state.model, state.batch)
