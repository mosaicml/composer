# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from composer.algorithms.seq_length_warmup import SeqLengthWarmup, set_batch_sequence_length
from composer.core import Event, State
from composer.devices import DeviceCPU
from composer.loggers import Logger
from tests.common.datasets import (dummy_bert_lm_dataloader, dummy_gpt_lm_dataloader,
                                   dummy_text_classification_dataloader)
from tests.common.models import SimpleTransformerClassifier, configure_tiny_bert_hf_model, configure_tiny_gpt2_hf_model


def check_batch_truncation(before, after, length, preserve_end_of_sequence=False):
    before_lengths = [int(m.sum()) for m in before['attention_mask']]

    # Just make sure the lengths are correct
    for k in before.keys():

        assert k in after, 'No keys should be removed during sequence truncation.'

        assert before[k].shape[0] == after[k].shape[
            0], 'The batch size should not be changed during sequence truncation.'

        if before[k].ndim >= 2:

            assert after[k].shape[1] == min(before[k].shape[1], length), 'Incorrect sequence length after truncation.'

            if preserve_end_of_sequence:
                # The last valid token before truncation should still be the last valid token
                for seq_before, seq_after, before_length in zip(before[k], after[k], before_lengths):
                    assert seq_after[min(length, before_length) - 1] == seq_before[before_length - 1]

    for k in after.keys():
        assert k in before, 'No keys should be added during sequence truncation'


def check_batch_non_truncation(before, after, length):
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


def check_batch(before, after, length, truncate: bool, preserve_end_of_sequence: bool):
    if truncate:
        check_batch_truncation(before, after, length, preserve_end_of_sequence)
    else:
        check_batch_non_truncation(before, after, length)


def check_forward_backward(model, batch):
    model.zero_grad()
    output = model.forward(batch)
    output['loss'].backward()


@pytest.mark.parametrize('model, dataloader', [
    (configure_tiny_bert_hf_model, dummy_bert_lm_dataloader),
    (configure_tiny_gpt2_hf_model, dummy_gpt_lm_dataloader),
    (pytest.param(
        SimpleTransformerClassifier,
        dummy_text_classification_dataloader,
        marks=pytest.mark.xfail(reason='Gated Linear Units does not currently support non-HuggingFace models'))),
])
@pytest.mark.parametrize('truncate,preserve_end_of_sequence', [(True, True), (True, False), (False, False)])
class TestSeqLengthWarmup:

    @pytest.mark.parametrize('curr_seq_length', [8, 64])
    def test_functional(self, model, dataloader, curr_seq_length: int, truncate: bool, preserve_end_of_sequence: bool):
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
        batch_before = next(iter(dataloader))
        batch_after = set_batch_sequence_length(deepcopy(batch_before), curr_seq_length, truncate,
                                                preserve_end_of_sequence)

        check_batch(batch_before, batch_after, curr_seq_length, truncate, preserve_end_of_sequence)
        check_forward_backward(state.model, batch_after)

    def test_algorithm(self, model, dataloader, empty_logger: Logger, truncate: bool, preserve_end_of_sequence: bool):
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

        # Synthetic dataset has a size of 2 batches per epoch (max duration = 1ep)
        seq_length_warmup = SeqLengthWarmup(duration=0.5,
                                            min_seq_length=8,
                                            max_seq_length=16,
                                            truncate=truncate,
                                            preserve_end_of_sequence=preserve_end_of_sequence)
        seq_length_warmup.apply(Event.INIT, state, empty_logger)

        batch_before = next(iter(dataloader))
        state.batch = deepcopy(batch_before)
        seq_length_warmup.apply(Event.AFTER_DATALOADER, state, empty_logger)

        # At this point, we should see the MINIMUM sequence length after truncation
        check_batch(batch_before, state.batch, seq_length_warmup.min_seq_length, truncate, preserve_end_of_sequence)
        check_forward_backward(state.model, state.batch)

        # Note: max duration is 1 epoch
        state.timestamp = state.timestamp.to_next_batch(samples=state.batch['input_ids'].shape[0])
        batch_before = next(iter(dataloader))
        state.batch = deepcopy(batch_before)
        seq_length_warmup.apply(Event.AFTER_DATALOADER, state, empty_logger)

        # At this point, we should see the MAXIMUM sequence length after truncation
        check_batch(batch_before, state.batch, seq_length_warmup.max_seq_length, truncate, preserve_end_of_sequence)
        check_forward_backward(state.model, state.batch)
