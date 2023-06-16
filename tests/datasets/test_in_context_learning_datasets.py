# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import random
from pathlib import Path

import pytest
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from composer import Evaluator
from composer.core import DataSpec
from composer.datasets.in_context_learning_evaluation import (_get_fewshot_sample_idxs, _make_padded_input,
                                                              get_icl_task_dataloader)
from composer.loggers import InMemoryLogger
from composer.metrics import (InContextLearningLMAccuracy, InContextLearningMultipleChoiceAccuracy,
                              InContextLearningQAAccuracy)
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, reproducibility
from tests.common import device, world_size


def test_fewshot_sample_idxs():
    rng = random.Random(1234)

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=4, sample_idx=4, rng=rng)
    assert fewshot_idxs == set([0, 1, 2, 3])

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=5, sample_idx=4, rng=rng)
    assert fewshot_idxs == set([0, 1, 2, 3])

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=500, sample_idx=4, rng=rng)
    assert fewshot_idxs == set([0, 1, 2, 3])

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=10, num_fewshot=7, sample_idx=4, rng=rng)
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


def test_batch_padding_logic(tiny_gpt2_tokenizer):
    continuation = tiny_gpt2_tokenizer(' dog' * 2000)['input_ids']
    context = tiny_gpt2_tokenizer(' cat' * 2000)['input_ids']
    _, continuation_spans = _make_padded_input(context, continuation, 2048, tiny_gpt2_tokenizer.eos_token_id)
    # the context (of len 2000) gets clipped to len 48 so that the whole continuation can fit
    assert continuation_spans[0] == 48 and continuation_spans[-1] == 2047


@pytest.mark.parametrize('padding_side', ['left', 'right', 'middle'])
def test_make_padding(tiny_gpt2_tokenizer, padding_side):
    context = tiny_gpt2_tokenizer(' cat' * 2000)['input_ids']
    padding_id = tiny_gpt2_tokenizer.eos_token_id

    error_context = contextlib.nullcontext() if padding_side in {'left', 'right'} else pytest.raises(ValueError)

    with error_context:
        input_ids, _ = _make_padded_input(context, [], 2048, padding_id, padding_side=padding_side)

        if padding_side == 'left':
            assert input_ids[0] == tiny_gpt2_tokenizer.eos_token_id
            assert input_ids[48:].tolist() == context
        elif padding_side == 'right':
            assert input_ids[-1] == tiny_gpt2_tokenizer.eos_token_id
            assert input_ids[:-48].tolist() == context


@pytest.mark.parametrize('dataset_uri', ['mmlu_small.jsonl'])
def test_mc_task_dataloader_subcategories(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 8
    seqlen = 2048
    dls = get_icl_task_dataloader('multiple_choice',
                                  dataset_uri,
                                  tokenizer,
                                  batch_size,
                                  max_seq_len=seqlen,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=2,
                                  prompt_string='The following are multiple choice questions (with answers).\n',
                                  example_delimiter='\n',
                                  continuation_delimiter='Answer: ',
                                  destination_path=str(tmp_path / 'icl.jsonl'),
                                  has_categories=True)
    assert isinstance(dls, dict)

    assert 'computer_security' in dls and 'human_aging' in dls
    dl = dls['computer_security']
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())
    assert dl.dataloader.__len__() == 4
    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    min_idx = min(batch['continuation_indices'][0]).item()
    max_idx = max(batch['continuation_indices'][0]).item()
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ' A'


@pytest.mark.parametrize('dataset_uri', [
    'pubmed_sm.jsonl',
])
def test_lm_task_dataloader_extra_space(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_icl_task_dataloader('language_modeling',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=10,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter=' ',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())

    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    min_idx = min(batch['continuation_indices'][0]).item()
    max_idx = max(batch['continuation_indices'][0]).item()
    assert '  ' not in tokenizer.decode(batch['input_ids'][0][0:max_idx + 1])
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ' yes'


@pytest.mark.parametrize('dataset_uri', [
    'lambada_small.jsonl',
])
def test_lm_task_dataloader(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_icl_task_dataloader('language_modeling',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=0,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter='',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())

    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    min_idx = min(batch['continuation_indices'][0]).item()
    max_idx = max(batch['continuation_indices'][0]).item()
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ' glen'


@pytest.mark.parametrize('dataset_uri', ['winograd_small.jsonl'])
def test_schema_task_dataloader(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_icl_task_dataloader('schema',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=1,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter='',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)
    batch = next(dl.dataloader._get_iterator())

    choices_per_question = 2
    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    assert 'gold_indices' in batch
    assert isinstance(batch['gold_indices'], list) and len(batch['gold_indices']) == batch_size // choices_per_question
    assert 'choice_groupings' in batch
    assert isinstance(batch['choice_groupings'], list) and len(
        batch['choice_groupings']) == batch_size // choices_per_question

    min_idx = min(batch['continuation_indices'][0]).item()
    max_idx = max(batch['continuation_indices'][0]).item()
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ' feared violence.'


@pytest.mark.parametrize('dataset_uri', ['winograd_small.jsonl'])
def test_schema_task_dataloader_sentpiece_tokenizer(dataset_uri, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b', use_fast=False)
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_icl_task_dataloader('schema',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=1,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter=' ',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)
    batch = next(dl.dataloader._get_iterator())

    choices_per_question = 2
    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    assert 'gold_indices' in batch
    assert isinstance(batch['gold_indices'], list) and len(batch['gold_indices']) == batch_size // choices_per_question
    assert 'choice_groupings' in batch
    assert isinstance(batch['choice_groupings'], list) and len(
        batch['choice_groupings']) == batch_size // choices_per_question

    max_idx = max(batch['continuation_indices'][0]).item()
    assert tokenizer.decode(
        batch['input_ids'][0][0:max_idx + 1]
    ) == "<s>Paul tried to call George on the phone, but George wasn't available. \nThe city councilmen refused the demonstrators a permit because the city councilmen feared violence."


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 1])
def test_lm_task_dataloader_opt_tokenizer(dataset_uri, num_fewshot, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m', use_fast=False)
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_icl_task_dataloader('language_modeling',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=num_fewshot,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter='',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())

    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    min_idx = min(batch['continuation_indices'][0]).item()
    max_idx = max(batch['continuation_indices'][0]).item()
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ' glen'
    assert tokenizer.decode(batch['input_ids'][0][0:min_idx]).startswith('</s>')
    assert tokenizer.decode(batch['input_ids'][0][0:min_idx]).count('</s>') == 1


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 1])
def test_mc_task_dataloader_opt_tokenizer(dataset_uri, num_fewshot, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m', use_fast=False)

    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 4
    seqlen = 2048
    dl = get_icl_task_dataloader('multiple_choice',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=num_fewshot,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter=': ',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())

    choices_per_question = 2
    assert dl.get_num_samples_in_batch(batch) == 2
    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    assert 'gold_indices' in batch
    assert isinstance(batch['gold_indices'], list) and len(batch['gold_indices']) == batch_size // choices_per_question
    assert 'choice_groupings' in batch
    assert isinstance(batch['choice_groupings'], list) and len(
        batch['choice_groupings']) == batch_size // choices_per_question

    min_idx = min(batch['continuation_indices'][0]).item()
    max_idx = max(batch['continuation_indices'][0]).item()
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ' Pour it onto a plate'
    assert tokenizer.decode(batch['input_ids'][0][0:min_idx]).startswith('</s>')
    assert tokenizer.decode(batch['input_ids'][0][0:min_idx]).count('</s>') == 1


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 1])
def test_mc_split_batch(dataset_uri, num_fewshot, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m', use_fast=False)

    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 4
    seqlen = 2048
    dl = get_icl_task_dataloader('multiple_choice',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=num_fewshot,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter=': ',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())
    choices_per_question = 2
    real_microbatch_size = batch_size // 2
    logical_microbatch_size = real_microbatch_size // choices_per_question
    microbatches = dl.split_batch(batch, logical_microbatch_size)
    assert len(microbatches) == 2
    for i, microbatch in enumerate(microbatches):
        assert dl.get_num_samples_in_batch(microbatch) == 1
        assert 'input_ids' in microbatch
        assert tuple(microbatch['input_ids'].shape) == (real_microbatch_size, seqlen)
        assert 'attention_mask' in microbatch
        assert tuple(microbatch['attention_mask'].shape) == (real_microbatch_size, seqlen)
        assert 'continuation_indices' in microbatch
        assert isinstance(microbatch['continuation_indices'], list) and len(
            microbatch['continuation_indices']) == real_microbatch_size
        assert 'mode' in microbatch
        assert microbatch['mode'] == 'icl_task'
        assert 'gold_indices' in microbatch
        assert isinstance(microbatch['gold_indices'], list) and len(
            microbatch['gold_indices']) == real_microbatch_size // choices_per_question
        assert 'choice_groupings' in microbatch
        assert isinstance(microbatch['choice_groupings'], list) and len(
            microbatch['choice_groupings']) == real_microbatch_size // choices_per_question

        min_idx = min(microbatch['continuation_indices'][0]).item()
        max_idx = max(microbatch['continuation_indices'][0]).item()
        if i == 0:
            assert tokenizer.decode(microbatch['input_ids'][0][min_idx:max_idx + 1]) == ' Pour it onto a plate'
        elif i == 1:
            assert tokenizer.decode(
                microbatch['input_ids'][0][min_idx:max_idx +
                                           1]) == ' Weld the metal together to get it to stay firmly in place'
        assert tokenizer.decode(microbatch['input_ids'][0][0:min_idx]).startswith('</s>')
        assert tokenizer.decode(microbatch['input_ids'][0][0:min_idx]).count('</s>') == 1


@pytest.mark.parametrize('dataset_uri', ['triviaqa_small.jsonl'])
def test_qa_split_batch(dataset_uri, tmp_path):
    pytest.importorskip('datasets')
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

    tmp_path_to_broadcast = str(os.path.abspath(tmp_path))
    gathered_paths = dist.all_gather_object(tmp_path_to_broadcast)
    dl = get_icl_task_dataloader(
        'question_answering',
        dataset_uri,
        tokenizer,
        8,
        max_seq_len=1024,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=0,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter=': ',
        destination_path=str(Path(gathered_paths[0]) / 'icl.jsonl'),
    )

    assert isinstance(dl, DataSpec)  # pyright

    batch = next(iter(dl.dataloader))
    split_batch = dl.split_batch(batch, 6)

    assert len(split_batch) == 2
    split1 = split_batch[0]
    split2 = split_batch[1]

    assert split1['input_ids'].shape[0] == 6
    assert split2['input_ids'].shape[0] == 2

    assert split1['attention_mask'].shape[0] == 6
    assert split2['attention_mask'].shape[0] == 2

    assert isinstance(split1['mode'], str)
    assert isinstance(split2['mode'], str)

    assert len(split1['labels']) == 6
    assert len(split2['labels']) == 2
    assert all(isinstance(v, list) for v in split1['labels'] + split2['labels'])

    assert isinstance(split1['generation_length'], int)
    assert isinstance(split2['generation_length'], int)

    assert isinstance(split1['generation_kwargs'], dict)
    assert isinstance(split2['generation_kwargs'], dict)


@pytest.mark.parametrize('dataset_uri', ['triviaqa_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 1, 2])
@pytest.mark.parametrize('prompt_string', ['I am a prompt', ''])
def test_qa_task_dataloader(dataset_uri, tiny_gpt2_tokenizer, tmp_path, num_fewshot, prompt_string):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    # empirical number from the small test dataset
    maximum_answer_length = 9
    dl = get_icl_task_dataloader('question_answering',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=num_fewshot,
                                 prompt_string=prompt_string,
                                 example_delimiter='\n',
                                 question_prelimiter='Q: ',
                                 continuation_delimiter='\nA:',
                                 destination_path=str(tmp_path / f'icl_{num_fewshot}.jsonl'))
    assert isinstance(dl, DataSpec)

    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())

    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen - maximum_answer_length)
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen - maximum_answer_length)
    assert batch['mode'] == 'generate'
    # the maximum generation length from the small test data
    assert batch['generation_length'] == maximum_answer_length
    assert all(item[0] == tokenizer.eos_token_id for item in batch['input_ids'])

    decoded_batch = tokenizer.batch_decode(batch['input_ids'])
    assert all([item.count('Q: ') == num_fewshot + 1 for item in decoded_batch])
    assert all([item.count('\nA:') == num_fewshot + 1 for item in decoded_batch])

    if len(prompt_string) > 0:
        assert all([item.count('I am a prompt') == 1 for item in decoded_batch])

    assert batch['labels'] == [['David Seville'], ['Scorpio', 'Skorpio']]

    assert decoded_batch[0].endswith('Q: Who was the man behind The Chipmunks?\nA:')
    assert decoded_batch[1].endswith('Q: What star sign is Jamie Lee Curtis?\nA:')


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonl'])
def test_mc_task_dataloader(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_icl_task_dataloader('multiple_choice',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=1,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter=': ',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    batch = next(dl.dataloader._get_iterator())

    choices_per_question = 2
    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'
    assert 'gold_indices' in batch
    assert isinstance(batch['gold_indices'], list) and len(batch['gold_indices']) == batch_size // choices_per_question
    assert 'choice_groupings' in batch
    assert isinstance(batch['choice_groupings'], list) and len(
        batch['choice_groupings']) == batch_size // choices_per_question

    min_idx = min(batch['continuation_indices'][0]).item()
    max_idx = max(batch['continuation_indices'][0]).item()
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ' Pour it onto a plate'


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 5])
@device('gpu')
def test_lm_task_evaluation(device, dataset_uri, num_fewshot, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    dl = get_icl_task_dataloader(
        'language_modeling',
        dataset_uri,
        tokenizer,
        2,
        max_seq_len=2048,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=num_fewshot,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter='',
        destination_path=str(tmp_path / 'icl.jsonl'),
    )

    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningLMAccuracy'])

    config = transformers.AutoConfig.from_pretrained('EleutherAI/gpt-neo-125M')
    model = transformers.AutoModelForCausalLM.from_config(config)
    model = HuggingFaceModel(
        model=model,
        tokenizer=None,
        eval_metrics=[InContextLearningLMAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ep', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator, subset_num_batches=2)
    assert 'metrics/lambada/InContextLearningLMAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/lambada/InContextLearningLMAccuracy'][0][1].item() == 0


@pytest.mark.parametrize('dataset_uri', ['winograd_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 5])
@pytest.mark.filterwarnings(r'ignore:Cannot split .* of length.*:UserWarning')
def test_schema_task_evaluation(num_fewshot, dataset_uri, tiny_gpt2_tokenizer, tmp_path, tiny_gpt2_model):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    dl = get_icl_task_dataloader(
        'schema',
        dataset_uri,
        tokenizer,
        8,
        max_seq_len=1024,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=num_fewshot,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter=': ',
        destination_path=str(tmp_path / 'icl.jsonl'),
    )

    evaluator = Evaluator(label='winograd', dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])

    model = HuggingFaceModel(
        model=tiny_gpt2_model,
        tokenizer=tokenizer,
        eval_metrics=[InContextLearningMultipleChoiceAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/winograd/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/winograd/InContextLearningMultipleChoiceAccuracy'][0][1].item() > 0
    num_samples = 0
    with open(dataset_uri) as f:
        for _ in f:
            num_samples += 1
    assert trainer.state.eval_metrics['winograd']['InContextLearningMultipleChoiceAccuracy'].total == num_samples


@pytest.mark.parametrize('dataset_uri', ['mmlu_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 5])
@device('gpu')
@world_size(1, 2)
@pytest.mark.filterwarnings(r'ignore:Cannot split .* of length.*:UserWarning')
def test_mc_task_evaluation_subcategories(device, world_size, dataset_uri, num_fewshot, tiny_gpt2_model,
                                          tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    tmp_path_to_broadcast = str(os.path.abspath(tmp_path))
    gathered_paths = dist.all_gather_object(tmp_path_to_broadcast)
    dls = get_icl_task_dataloader('multiple_choice',
                                  dataset_uri,
                                  tokenizer,
                                  8,
                                  max_seq_len=1024,
                                  pad_tok_id=tokenizer.eos_token_id,
                                  num_fewshot=num_fewshot,
                                  prompt_string='',
                                  example_delimiter='\n',
                                  continuation_delimiter=': ',
                                  destination_path=str(Path(gathered_paths[0]) / 'icl.jsonl'),
                                  has_categories=True)

    assert isinstance(dls, dict)
    evaluators = [
        Evaluator(label='mmlu/' + k, dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])
        for k, dl in dls.items()
    ]

    model = HuggingFaceModel(
        model=tiny_gpt2_model,
        tokenizer=tiny_gpt2_tokenizer,
        eval_metrics=[InContextLearningMultipleChoiceAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluators)
    assert 'metrics/mmlu/computer_security/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    assert 'metrics/mmlu/human_aging/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/mmlu/computer_security/InContextLearningMultipleChoiceAccuracy'][0][1].item(
    ) > 0
    total = trainer.state.eval_metrics['mmlu/computer_security']['InContextLearningMultipleChoiceAccuracy'].total
    dist.all_reduce(total)  # type: ignore
    assert total.item() == 8  # type: ignore
    total = trainer.state.eval_metrics['mmlu/human_aging']['InContextLearningMultipleChoiceAccuracy'].total
    dist.all_reduce(total)  # type: ignore
    assert total.item() == 7  # type: ignore


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonl', 'hellaswag_small.jsonl'])
@device('gpu')
@pytest.mark.parametrize('num_fewshot', [0, 5])
def test_mc_task_evaluation(device, num_fewshot, dataset_uri, tiny_gpt2_tokenizer, tmp_path, tiny_gpt2_model):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer

    # seed because the fewshot selection is currently unseeded
    reproducibility.seed_all(1234)
    dl = get_icl_task_dataloader(
        'multiple_choice',
        dataset_uri,
        tokenizer,
        8,
        max_seq_len=1024,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=num_fewshot,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter=': ',
        destination_path=str(tmp_path / 'icl.jsonl'),
    )

    evaluator = Evaluator(label='mc', dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])

    model = HuggingFaceModel(
        model=tiny_gpt2_model,
        tokenizer=tiny_gpt2_tokenizer,
        eval_metrics=[InContextLearningMultipleChoiceAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/mc/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/mc/InContextLearningMultipleChoiceAccuracy'][0][1].item() > 0
    num_samples = 0
    with open(dataset_uri) as f:
        for _ in f:
            num_samples += 1
    assert trainer.state.eval_metrics['mc']['InContextLearningMultipleChoiceAccuracy'].total == num_samples


@pytest.mark.parametrize('dataset_uri', ['triviaqa_small.jsonl'])
@device('gpu')
@world_size(1, 2)
@pytest.mark.parametrize('num_fewshot', [0, 5])
def test_qa_task_evaluation_opt_tokenizer(device, world_size, num_fewshot, dataset_uri, tmp_path):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

    tmp_path_to_broadcast = str(os.path.abspath(tmp_path))
    gathered_paths = dist.all_gather_object(tmp_path_to_broadcast)
    dl = get_icl_task_dataloader(
        'question_answering',
        dataset_uri,
        tokenizer,
        2,
        max_seq_len=1024,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=num_fewshot,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter=': ',
        destination_path=str(Path(gathered_paths[0]) / 'icl.jsonl'),
    )

    evaluator = Evaluator(label='triviaqa', dataloader=dl, metric_names=['InContextLearningQAAccuracy'])
    model = HuggingFaceModel(
        model=AutoModelForCausalLM.from_pretrained('facebook/opt-125m'),
        tokenizer=tokenizer,
        eval_metrics=[InContextLearningQAAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)

    trainer.eval(eval_dataloader=evaluator, subset_num_batches=2)
    assert 'metrics/triviaqa/InContextLearningQAAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/triviaqa/InContextLearningQAAccuracy'][0][1].item() == 0


@pytest.mark.parametrize('dataset_uri', ['triviaqa_small.jsonl'])
@device('gpu')
@world_size(1, 2)
@pytest.mark.parametrize('num_fewshot', [0, 5])
def test_qa_task_evaluation(device, world_size, num_fewshot, dataset_uri, tiny_gpt2_tokenizer, tiny_gpt2_model,
                            tmp_path):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    tmp_path_to_broadcast = str(os.path.abspath(tmp_path))
    gathered_paths = dist.all_gather_object(tmp_path_to_broadcast)
    dl = get_icl_task_dataloader(
        'question_answering',
        dataset_uri,
        tokenizer,
        2,
        max_seq_len=1024,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=num_fewshot,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter=': ',
        destination_path=str(Path(gathered_paths[0]) / 'icl.jsonl'),
    )

    evaluator = Evaluator(label='triviaqa', dataloader=dl, metric_names=['InContextLearningQAAccuracy'])

    model = HuggingFaceModel(
        model=tiny_gpt2_model,
        tokenizer=tiny_gpt2_tokenizer,
        eval_metrics=[InContextLearningQAAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)

    trainer.eval(eval_dataloader=evaluator, subset_num_batches=2)
    assert 'metrics/triviaqa/InContextLearningQAAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/triviaqa/InContextLearningQAAccuracy'][0][1].item() == 0


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonl'])
def test_lm_spacing_dataloader(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 1
    seqlen = 2048
    dl = get_icl_task_dataloader('language_modeling',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=1,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter=' UNIQUE ',
                                 destination_path=str(tmp_path / 'icl.jsonl'))
    assert isinstance(dl, DataSpec)
    assert isinstance(dl.dataloader, DataLoader)  # pyright
    first_batch = next(dl.dataloader._get_iterator())
    second_batch = next(dl.dataloader._get_iterator())

    first_batch_text = tokenizer.decode(first_batch['input_ids'][0], skip_special_tokens=True)
    second_batch_text = tokenizer.decode(second_batch['input_ids'][0], skip_special_tokens=True)

    first_batch_without_last_word = ' '.join(first_batch_text.split(' ')[:-1])
    second_batch_without_last_word = ' '.join(second_batch_text.split(' ')[:-1])

    assert first_batch_without_last_word.endswith(' UNIQUE')
    assert second_batch_without_last_word.endswith(' UNIQUE')

    assert first_batch_without_last_word.count(' UNIQUE ') == 1
    assert second_batch_without_last_word.count(' UNIQUE ') == 1
