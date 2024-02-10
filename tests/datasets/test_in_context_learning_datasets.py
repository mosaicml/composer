# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import random
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from composer import Evaluator
from composer.core import DataSpec

# isort: off
from composer.datasets.in_context_learning_evaluation import (
    InContextLearningDataset,
    _get_continuation_span,
    _get_fewshot_sample_idxs,
    _make_padded_input,
    _tokenizer_needs_prefix_space,
    _trim_context,
    strip_data,
)
# isort: on
from composer.datasets.utils import MultiTokenEOSCriteria
from composer.loggers import InMemoryLogger
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, reproducibility
from tests.common import device, world_size


def test_strip_data():
    data_to_strip = {'strip_data': '  boo!  \n', 'has_space': '  wa hoo!', 'end_space': 'yoohoo!  '}
    stripped_data = strip_data(data_to_strip)
    for k, v in stripped_data.items():
        assert k in data_to_strip
        assert not v[0].isspace()
        assert not v[-1].isspace()


@pytest.mark.skip(reason="Currently don't have a tokenizer that satisfies this test")
def test_tokenizer_needs_prefix_space_when_space_not_needed(tiny_gpt2_tokenizer):
    assert not _tokenizer_needs_prefix_space(tiny_gpt2_tokenizer)


def test_tokenizer_needs_prefix_space_when_space_needed():
    transformers = pytest.importorskip('transformers')
    tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/opt-125m',
                                                           use_fast=False)  # type: ignore reportUnboundVariable
    assert _tokenizer_needs_prefix_space(tokenizer)


def test_trim_context():
    context = [0] * 99 + [1] * 2037
    continuation = [2] * 10
    max_seq_len = 2048
    trimmed_context = _trim_context(context, continuation, max_seq_len=max_seq_len)
    assert len(trimmed_context) == 2038
    assert trimmed_context[0] == 0
    assert trimmed_context[1] == 1


def test_trim_context_no_continuation():
    context = [0] * 2048
    max_seq_len = 2048
    trimmed_context = _trim_context(context, [], max_seq_len=max_seq_len)
    assert len(trimmed_context) == 2048
    context = [0] * 3000 + [1]
    max_seq_len = 2048
    trimmed_context = _trim_context(context, [], max_seq_len=max_seq_len)
    assert len(trimmed_context) == 2048
    assert trimmed_context[-1] == 1


def test_get_continuation_span():
    context = [0] * 200
    continuation = [1] * 3
    cont_span = _get_continuation_span(context, continuation)
    assert torch.all(torch.eq(cont_span, torch.tensor([200, 201, 202])))
    continuation = [1]
    cont_span = _get_continuation_span(context, continuation)
    assert torch.all(torch.eq(cont_span, torch.tensor([200])))


@pytest.mark.parametrize('padding_side', ['left', 'right', 'middle'])
def test_make_padding(tiny_gpt2_tokenizer, padding_side):
    context = tiny_gpt2_tokenizer(' cat' * 2000)['input_ids']
    padding_id = tiny_gpt2_tokenizer.eos_token_id

    error_context = contextlib.nullcontext() if padding_side in {'left', 'right'} else pytest.raises(ValueError)

    with error_context:
        input_ids = _make_padded_input(context, [], 2048, padding_id, padding_side=padding_side)

        if padding_side == 'left':
            assert input_ids[0] == tiny_gpt2_tokenizer.eos_token_id
            assert input_ids[48:].tolist() == context
        elif padding_side == 'right':
            assert input_ids[-1] == tiny_gpt2_tokenizer.eos_token_id
            assert input_ids[:-48].tolist() == context


def test_batch_padding_logic_no_padding(tiny_gpt2_tokenizer):
    continuation = tiny_gpt2_tokenizer(' dog' * 2000)['input_ids']
    context = tiny_gpt2_tokenizer(' cat' * 2000)['input_ids']
    max_seq_len = 2048
    trimmed_context = _trim_context(context, continuation, max_seq_len)
    continuation_spans = _get_continuation_span(trimmed_context, continuation)
    padded_input = _make_padded_input(trimmed_context,
                                      continuation,
                                      max_seq_len,
                                      tiny_gpt2_tokenizer.pad_token_id,
                                      padding_side='right')
    assert continuation_spans[0] == 48 and continuation_spans[-1] == 2047
    assert len(padded_input) == 2048
    assert tiny_gpt2_tokenizer.pad_token_id not in padded_input


def test_batch_padding_logic_with_padding(tiny_gpt2_tokenizer):
    continuation = tiny_gpt2_tokenizer(' dog' * 200)['input_ids']
    context = tiny_gpt2_tokenizer(' cat' * 200)['input_ids']
    max_seq_len = 2048
    trimmed_context = _trim_context(context, continuation, max_seq_len)
    continuation_spans = _get_continuation_span(trimmed_context, continuation)
    padded_input = _make_padded_input(trimmed_context,
                                      continuation,
                                      max_seq_len,
                                      tiny_gpt2_tokenizer.pad_token_id,
                                      padding_side='right')
    assert continuation_spans[0] == 200 and continuation_spans[-1] == 399
    assert len(padded_input) == 2048
    assert padded_input[-1] == tiny_gpt2_tokenizer.pad_token_id


def test_fewshot_sample_idxs():
    rng = random.Random(1234)

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=4, example_idx=4, rng=rng)
    assert fewshot_idxs == {0, 1, 2, 3}

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=5, example_idx=4, rng=rng)
    assert fewshot_idxs == {0, 1, 2, 3}

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=500, example_idx=4, rng=rng)
    assert fewshot_idxs == {0, 1, 2, 3}

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=10, num_fewshot=7, example_idx=4, rng=rng)
    assert len(fewshot_idxs) == 7 and 4 not in fewshot_idxs


def test_fewshot_sample_idxs_randomness():
    dataset_size = 10000
    num_fewshot = 5

    rng_1_seed_1234 = random.Random(1234)
    rng_2_seed_1234 = random.Random(1234)
    rng_3_seed_11 = random.Random(11)

    rng_1_sample_1 = _get_fewshot_sample_idxs(dataset_size, num_fewshot, 1, rng_1_seed_1234)
    rng_2_sample_1 = _get_fewshot_sample_idxs(dataset_size, num_fewshot, 1, rng_2_seed_1234)
    rng_3_sample_1 = _get_fewshot_sample_idxs(dataset_size, num_fewshot, 1, rng_3_seed_11)

    assert rng_1_sample_1 == rng_2_sample_1
    assert rng_1_sample_1 != rng_3_sample_1

    rng_1_sample_2 = _get_fewshot_sample_idxs(dataset_size, num_fewshot, 2, rng_1_seed_1234)
    rng_2_sample_2 = _get_fewshot_sample_idxs(dataset_size, num_fewshot, 2, rng_2_seed_1234)
    rng_3_sample_2 = _get_fewshot_sample_idxs(dataset_size, num_fewshot, 2, rng_3_seed_11)

    assert rng_1_sample_2 == rng_2_sample_2
    assert rng_1_sample_2 != rng_3_sample_2


@pytest.mark.filterwarnings(
    r'ignore:The repository for mosaicml/test_dataset contains custom code which must*:FutureWarning')
def test_update_generation_kwargs(tiny_gpt2_tokenizer, tmp_path):
    tokenizer = tiny_gpt2_tokenizer
    seqlen = 2048
    num_fewshot = 0
    prompt_string = ''
    hf_loading_vars = {
        'split': 'test',
        'name': 'invoker',
    }
    hf_parsing_map = {'context': ['quas', 'wex', 'exort'], 'answer': ['spell']}
    gen_kwargs = {'test_arg1': 1, 'test_arg2': 2}

    dl = InContextLearningDataset(dataset_uri='hf://mosaicml/test_dataset',
                                  tokenizer=tokenizer,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  fewshot_random_seed=1,
                                  prompt_string=prompt_string,
                                  example_delimiter='\n',
                                  prelimiter='Orbs: ',
                                  continuation_delimiter='\nSpell:',
                                  destination_path=str(tmp_path / 'test_dataset_lm_juggernaut.jsonl'),
                                  hf_loading_vars=hf_loading_vars,
                                  hf_parsing_map=hf_parsing_map,
                                  generation_kwargs=gen_kwargs)
    assert dl.base_batch['generation_kwargs'] == {'test_arg1': 1, 'test_arg2': 2}


def test_stop_sequences_criteria(tiny_gpt2_tokenizer):
    pytest.importorskip('transformers')
    eos_criteria = MultiTokenEOSCriteria('\n\n', tiny_gpt2_tokenizer, 2)
    seq1 = tiny_gpt2_tokenizer('Dogs are furry')['input_ids']
    seq2 = tiny_gpt2_tokenizer('Dogs are furry\n\n')['input_ids']
    seq1 = [tiny_gpt2_tokenizer.pad_token_id] * (len(seq2) - len(seq1)) + seq1
    input_ids = torch.LongTensor([seq1, seq2])
    assert not eos_criteria(input_ids, None)  # pyright: ignore[reportGeneralTypeIssues]

    eos_criteria = MultiTokenEOSCriteria('\n\n', tiny_gpt2_tokenizer, 2)
    seq1 = tiny_gpt2_tokenizer('Dogs are furry\n\n')['input_ids']
    seq2 = tiny_gpt2_tokenizer('Dogs are furry\n\n')['input_ids']
    input_ids = torch.LongTensor([seq1, seq2])
    assert eos_criteria(input_ids, None)  # pyright: ignore[reportGeneralTypeIssues]


def test_stop_sequences_criteria_sentencepiece(tiny_llama_tokenizer):
    pytest.importorskip('datasets')

    tokenizer = tiny_llama_tokenizer
    eos_criteria = MultiTokenEOSCriteria('\n\n', tokenizer, 2)
    seq1 = tokenizer('\n\nDogs')['input_ids']  # check to make sure starting with the stop sequence doesnt break it
    seq2 = tokenizer('Dogs are furry\n\n')['input_ids']
    seq1 = [tokenizer.eos_token_id] * (len(seq2) - len(seq1)) + seq1
    input_ids = torch.LongTensor([seq1, seq2])
    assert not eos_criteria(input_ids, None)  # pyright: ignore[reportGeneralTypeIssues]

    eos_criteria = MultiTokenEOSCriteria('\n\n', tokenizer, 2)
    seq1 = tokenizer('Dogs are furry\n\n')['input_ids']
    seq2 = tokenizer('Dogs are furry\n\n')['input_ids']
    input_ids = torch.LongTensor([seq1, seq2])
    assert eos_criteria(input_ids, None)  # pyright: ignore[reportGeneralTypeIssues]


@pytest.mark.filterwarnings(
    r'ignore:The repository for mosaicml/test_dataset contains custom code which must*:FutureWarning')
def test_update_generation_kwargs_no_kwargs(tiny_gpt2_tokenizer, tmp_path):
    tokenizer = tiny_gpt2_tokenizer
    seqlen = 2048
    num_fewshot = 0
    prompt_string = ''
    hf_loading_vars = {
        'split': 'test',
        'name': 'invoker',
    }
    hf_parsing_map = {'context': ['quas', 'wex', 'exort'], 'answer': ['spell']}

    dl = InContextLearningDataset(dataset_uri='hf://mosaicml/test_dataset',
                                  tokenizer=tokenizer,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  fewshot_random_seed=1,
                                  prompt_string=prompt_string,
                                  example_delimiter='\n',
                                  prelimiter='Orbs: ',
                                  continuation_delimiter='\nSpell:',
                                  destination_path=str(tmp_path / 'test_dataset_lm_juggernaut.jsonl'),
                                  hf_loading_vars=hf_loading_vars,
                                  hf_parsing_map=hf_parsing_map)
    assert not 'generation_kwargs' in dl.base_batch


@pytest.mark.filterwarnings(
    r'ignore:The repository for mosaicml/test_dataset contains custom code which must*:FutureWarning')
def test_construct_context(tiny_gpt2_tokenizer, tmp_path):
    tokenizer = tiny_gpt2_tokenizer
    seqlen = 2048
    num_fewshot = 0
    prompt_string = ''
    hf_loading_vars = {
        'split': 'test',
        'name': 'invoker',
    }
    hf_parsing_map = {'context': ['quas', 'wex', 'exort'], 'answer': ['spell']}

    dl = InContextLearningDataset(dataset_uri='hf://mosaicml/test_dataset',
                                  tokenizer=tokenizer,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  fewshot_random_seed=1,
                                  prompt_string=prompt_string,
                                  example_delimiter='\n',
                                  prelimiter='Orbs: ',
                                  continuation_delimiter='\nSpell: ',
                                  destination_path=str(tmp_path / 'test_dataset_lm_juggernaut.jsonl'),
                                  hf_loading_vars=hf_loading_vars,
                                  hf_parsing_map=hf_parsing_map)
    constructed_context = dl.construct_context({'context': 'quas quas exort', 'answer': 'ice wall'})
    assert constructed_context == 'Orbs: quas quas exort\nSpell: '
    constructed_context = dl.construct_context({'context': 'quas quas exort', 'answer': 'ice wall'}, add_answer=True)
    assert constructed_context == 'Orbs: quas quas exort\nSpell: ice wall'
    constructed_context = dl.construct_context({
        'context': 'quas quas exort',
        'answer': 'ice wall'
    },
                                               preceding_text='The harsh White Waste beckons!',
                                               add_answer=True)
    assert constructed_context == '\nOrbs: quas quas exort\nSpell: ice wall'


@pytest.mark.filterwarnings(
    r'ignore:The repository for mosaicml/test_dataset contains custom code which must*:FutureWarning')
def test_get_answer_from_example(tiny_gpt2_tokenizer, tmp_path):
    tokenizer = tiny_gpt2_tokenizer
    seqlen = 2048
    num_fewshot = 0
    prompt_string = ''
    hf_loading_vars = {
        'split': 'test',
        'name': 'invoker',
    }
    hf_parsing_map = {'context': ['quas', 'wex', 'exort'], 'answer': ['spell']}

    dl = InContextLearningDataset(dataset_uri='hf://mosaicml/test_dataset',
                                  tokenizer=tokenizer,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  fewshot_random_seed=1,
                                  prompt_string=prompt_string,
                                  example_delimiter='\n',
                                  prelimiter='Orbs: ',
                                  continuation_delimiter='\nSpell:',
                                  destination_path=str(tmp_path / 'test_dataset_lm_juggernaut.jsonl'),
                                  hf_loading_vars=hf_loading_vars,
                                  hf_parsing_map=hf_parsing_map)
    answer = dl.get_answer_from_example({'context': 'wex exort exort', 'answer': 'alacrity'})
    assert answer == ' alacrity'


@pytest.mark.filterwarnings(
    r'ignore:The repository for mosaicml/test_dataset contains custom code which must*:FutureWarning')
def test_fix_eos_on_preamble(tmp_path):
    transformers = pytest.importorskip('transformers')
    tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/opt-125m',
                                                           use_fast=False)  # type: ignore reportUnboundVariable
    seqlen = 2048
    num_fewshot = 0
    prompt_string = ''
    hf_loading_vars = {
        'split': 'test',
        'name': 'invoker',
    }
    hf_parsing_map = {'context': ['quas', 'wex', 'exort'], 'answer': ['spell']}

    dl = InContextLearningDataset(dataset_uri='hf://mosaicml/test_dataset',
                                  tokenizer=tokenizer,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  fewshot_random_seed=1,
                                  prompt_string=prompt_string,
                                  example_delimiter='\n',
                                  prelimiter='Orbs: ',
                                  continuation_delimiter='\nSpell:',
                                  destination_path=str(tmp_path / 'test_dataset_lm_juggernaut.jsonl'),
                                  hf_loading_vars=hf_loading_vars,
                                  hf_parsing_map=hf_parsing_map)
    preamble = 'blah blah blah.'
    tokenized_preamble = tokenizer.encode(preamble)
    tokenized_preamble += [tokenizer.eos_token_id]
    fixed_preamble = dl._fix_eos_on_preamble(tokenized_preamble)
    assert tokenized_preamble[:-1] == fixed_preamble
    assert fixed_preamble[-1] != tokenizer.eos_token_id


@pytest.mark.filterwarnings(
    r'ignore:The repository for mosaicml/test_dataset contains custom code which must*:FutureWarning')
def test_tokenize_example_with_tokenize_labels(tiny_gpt2_tokenizer, tmp_path):
    tokenizer = tiny_gpt2_tokenizer
    seqlen = 2048
    num_fewshot = 0
    prompt_string = ''
    hf_loading_vars = {
        'split': 'test',
        'name': 'invoker',
    }
    hf_parsing_map = {'context': ['quas', 'wex', 'exort'], 'answer': ['spell']}

    dl = InContextLearningDataset(dataset_uri='hf://mosaicml/test_dataset',
                                  tokenizer=tokenizer,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  fewshot_random_seed=1,
                                  prompt_string=prompt_string,
                                  example_delimiter='\n',
                                  prelimiter='Orbs: ',
                                  continuation_delimiter='\nSpell: ',
                                  destination_path=str(tmp_path / 'test_dataset_lm_juggernaut.jsonl'),
                                  hf_loading_vars=hf_loading_vars,
                                  hf_parsing_map=hf_parsing_map,
                                  tokenize_labels=True)
    tokenized_example = dl.tokenize_example('What spell does this invoke? ', 'exort exort wex\nSpell: ',
                                            {'answer': ' Meatball'})
    tokenized_input = [2061, 4822, 857, 428, 26342, 30, 220, 1069, 419, 409, 419, 356, 87, 198, 31221, 25, 19145, 1894]
    assert tokenized_example['context'][:len(tokenized_input)].tolist() == tokenized_input
    assert tokenized_example['context'][-1] == tokenizer.eos_token_id
    assert type(tokenized_example['answer'][0]) == int
    assert len(tokenized_example['context']) == seqlen
    assert 'continuation_indices' in tokenized_example


@pytest.mark.filterwarnings(
    r'ignore:The repository for mosaicml/test_dataset contains custom code which must*:FutureWarning')
def test_tokenize_example_with_no_tokenize_labels(tiny_gpt2_tokenizer, tmp_path):
    tokenizer = tiny_gpt2_tokenizer
    seqlen = 2048
    num_fewshot = 0
    prompt_string = ''
    hf_loading_vars = {
        'split': 'test',
        'name': 'invoker',
    }
    hf_parsing_map = {'context': ['quas', 'wex', 'exort'], 'answer': ['spell']}

    dl = InContextLearningDataset(dataset_uri='hf://mosaicml/test_dataset',
                                  tokenizer=tokenizer,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  fewshot_random_seed=1,
                                  prompt_string=prompt_string,
                                  example_delimiter='\n',
                                  prelimiter='Orbs: ',
                                  continuation_delimiter='\nSpell: ',
                                  destination_path=str(tmp_path / 'test_dataset_lm_juggernaut.jsonl'),
                                  hf_loading_vars=hf_loading_vars,
                                  hf_parsing_map=hf_parsing_map,
                                  tokenize_labels=False)
    tokenized_example = dl.tokenize_example('What spell does this invoke? ', 'exort exort wex\nSpell: ',
                                            {'answer': ' Meatball'})
    tokenized_input = [2061, 4822, 857, 428, 26342, 30, 220, 1069, 419, 409, 419, 356, 87, 198, 31221, 25]
    assert tokenized_example['context'][:len(tokenized_input)].tolist() == tokenized_input
    assert tokenized_example['context'][-1] == tokenizer.eos_token_id
    assert len(tokenized_example['context']) == seqlen
    assert type(tokenized_example['answer']) == str
