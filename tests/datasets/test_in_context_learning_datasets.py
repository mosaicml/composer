# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from pathlib import Path

import pytest
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from composer.datasets.in_context_learning_evaluation import (_get_fewshot_sample_idxs, _make_padded_input,
                                                              get_dataloaders_with_category, make_evaluators)
from composer.loggers import InMemoryLogger
from composer.metrics import (InContextLearningLMAccuracy, InContextLearningMultipleChoiceAccuracy,
                              InContextLearningQAAccuracy)
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, reproducibility
from tests.common import device, world_size


def test_fewshot_sample_idxs():
    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=4, sample_idx=4)
    assert fewshot_idxs == set([0, 1, 2, 3])

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=5, sample_idx=4)
    assert fewshot_idxs == set([0, 1, 2, 3])

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=5, num_fewshot=500, sample_idx=4)
    assert fewshot_idxs == set([0, 1, 2, 3])

    fewshot_idxs = _get_fewshot_sample_idxs(dataset_size=10, num_fewshot=7, sample_idx=4)
    assert len(fewshot_idxs) == 7 and 4 not in fewshot_idxs


def test_batch_padding_logic(tiny_gpt2_tokenizer):
    continuation = tiny_gpt2_tokenizer(' dog' * 2000)['input_ids']
    context = tiny_gpt2_tokenizer(' cat' * 2000)['input_ids']
    _, continuation_spans = _make_padded_input(context, continuation, 2048, tiny_gpt2_tokenizer.eos_token_id)
    # the context (of len 2000) gets clipped to len 48 so that the whole continuation can fit
    assert continuation_spans[0] == 48 and continuation_spans[-1] == 2047


@pytest.mark.parametrize('dataset_uri', ['jeopardy_small.jsonl', 'wiki_people_small.jsonl'])
def test_lm_task_dataloader_jeopardy(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048

    if dataset_uri == 'jeopardy_small.jsonl':
        categories = ['science', 'world_history']
    else:
        categories = ['num_spouses', 'birth_day']

    dls = get_dataloaders_with_category('language_modeling',
                                        dataset_uri,
                                        categories,
                                        tokenizer,
                                        batch_size,
                                        max_seq_len=seqlen,
                                        pad_tok_id=tokenizer.eos_token_id,
                                        num_fewshot=4,
                                        prompt_string='',
                                        example_delimiter='\n',
                                        continuation_delimiter='. Answer: ',
                                        destination_path=str(tmp_path / 'icl.jsonl'))

    assert set(dls.keys()) == set(['', '/science', '/world_history']) or set(dls.keys()) == set(
        ['', '/num_spouses', '/birth_day'])
    assert isinstance(dls[''].dataloader, DataLoader)  # pyright
    batch = next(dls[''].dataloader._get_iterator())
    assert set(batch.keys()) == set(['input_ids', 'attention_mask', 'labels', 'continuation_indices', 'mode'])
    assert 'input_ids' in batch
    assert tuple(batch['input_ids'].shape) == (batch_size, seqlen)
    assert 'attention_mask' in batch
    assert tuple(batch['attention_mask'].shape) == (batch_size, seqlen)
    assert 'continuation_indices' in batch
    assert isinstance(batch['continuation_indices'], list) and len(batch['continuation_indices']) == batch_size
    assert 'mode' in batch
    assert batch['mode'] == 'icl_task'


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


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 2, 8])
def test_lm_task_dataloader(dataset_uri, num_fewshot, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_dataloaders_with_category('language_modeling',
                                       dataset_uri, [],
                                       tokenizer,
                                       batch_size,
                                       max_seq_len=seqlen,
                                       pad_tok_id=tokenizer.eos_token_id,
                                       num_fewshot=num_fewshot,
                                       prompt_string='',
                                       example_delimiter='\n',
                                       continuation_delimiter='',
                                       destination_path=str(tmp_path / 'icl.jsonl'))['']

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


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 1])
def test_lm_task_dataloader_opt_tokenizer(dataset_uri, num_fewshot, tmp_path):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m', use_fast=False)
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dl = get_dataloaders_with_category('language_modeling',
                                       dataset_uri, [],
                                       tokenizer,
                                       batch_size,
                                       max_seq_len=seqlen,
                                       pad_tok_id=tokenizer.eos_token_id,
                                       num_fewshot=num_fewshot,
                                       prompt_string='',
                                       example_delimiter='\n',
                                       continuation_delimiter='',
                                       destination_path=str(tmp_path / 'icl.jsonl'))['']

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
    batch_size = 2
    seqlen = 2048
    dl = get_dataloaders_with_category('multiple_choice',
                                       dataset_uri, [],
                                       tokenizer,
                                       batch_size,
                                       max_seq_len=seqlen,
                                       pad_tok_id=tokenizer.eos_token_id,
                                       num_fewshot=num_fewshot,
                                       prompt_string='',
                                       example_delimiter='\n',
                                       continuation_delimiter=': ',
                                       destination_path=str(tmp_path / 'icl.jsonl'))['']

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
    assert tokenizer.decode(batch['input_ids'][0][min_idx - 1:max_idx + 1]) == ': Pour it onto a plate'
    assert tokenizer.decode(batch['input_ids'][0][0:min_idx]).startswith('</s>')
    assert tokenizer.decode(batch['input_ids'][0][0:min_idx]).count('</s>') == 1


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
    dls = get_dataloaders_with_category('question_answering',
                                        dataset_uri, [],
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

    assert isinstance(dls[''].dataloader, DataLoader)  # pyright
    batch = next(dls[''].dataloader._get_iterator())

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


@pytest.mark.parametrize('dataset_uri', ['triviaqa_small_entities.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 1, 2])
@pytest.mark.parametrize('prompt_string', ['I am a prompt', ''])
def test_qa_task_dataloader_entities(dataset_uri, tiny_gpt2_tokenizer, tmp_path, num_fewshot, prompt_string):
    pytest.importorskip('datasets')

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    # empirical number from the small test dataset
    maximum_answer_length = 7
    dls = get_dataloaders_with_category('question_answering',
                                        dataset_uri, ['foo'],
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

    assert isinstance(dls[''].dataloader, DataLoader)  # pyright
    batch = next(dls[''].dataloader._get_iterator())

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
    dl = get_dataloaders_with_category('multiple_choice',
                                       dataset_uri, [],
                                       tokenizer,
                                       batch_size,
                                       max_seq_len=seqlen,
                                       pad_tok_id=tokenizer.eos_token_id,
                                       num_fewshot=1,
                                       prompt_string='',
                                       example_delimiter='\n',
                                       continuation_delimiter=': ',
                                       destination_path=str(tmp_path / 'icl.jsonl'))['']

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
    assert tokenizer.decode(batch['input_ids'][0][min_idx - 1:max_idx + 1]) == ': Pour it onto a plate'


@pytest.mark.parametrize('dataset_uri', ['piqa_small_categories.jsonl'])
def test_mc_task_dataloader_categories(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dls = get_dataloaders_with_category('multiple_choice',
                                        dataset_uri, ['foo', 'bar'],
                                        tokenizer,
                                        batch_size,
                                        max_seq_len=seqlen,
                                        pad_tok_id=tokenizer.eos_token_id,
                                        num_fewshot=1,
                                        prompt_string='',
                                        example_delimiter='\n',
                                        continuation_delimiter=': ',
                                        destination_path=str(tmp_path / 'icl.jsonl'))

    assert isinstance(dls[''].dataloader, DataLoader)  # pyright
    batch = next(dls[''].dataloader._get_iterator())
    assert set(dls.keys()) == set(['', '/foo', '/bar'])
    assert isinstance(dls[''].dataloader, DataLoader)  # pyright
    batch = next(dls[''].dataloader._get_iterator())
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
    assert tokenizer.decode(batch['input_ids'][0][min_idx - 1:max_idx + 1]) == ': Pour it onto a plate'


@pytest.mark.parametrize('dataset_uri', ['piqa_small_entities.jsonl'])
def test_mc_task_dataloader_entities(dataset_uri, tiny_gpt2_tokenizer, tmp_path):
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 2
    seqlen = 2048
    dls = get_dataloaders_with_category('multiple_choice',
                                        dataset_uri, ['foo'],
                                        tokenizer,
                                        batch_size,
                                        max_seq_len=seqlen,
                                        pad_tok_id=tokenizer.eos_token_id,
                                        num_fewshot=1,
                                        prompt_string='',
                                        example_delimiter='\n',
                                        continuation_delimiter=': ',
                                        destination_path=str(tmp_path / 'icl.jsonl'))

    assert isinstance(dls[''].dataloader, DataLoader)  # pyright
    batch = next(dls[''].dataloader._get_iterator())
    assert set(dls.keys()) == set(['', '/foo'])
    assert isinstance(dls[''].dataloader, DataLoader)  # pyright
    batch = next(dls[''].dataloader._get_iterator())
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
    assert tokenizer.decode(batch['input_ids'][0][min_idx - 1:max_idx + 1]) == ': Pour it onto a plate'


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [0, 5])
@device('gpu')
def test_lm_task_evaluation(device, dataset_uri, num_fewshot, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer

    evaluators = make_evaluators(
        'lambada',
        ['InContextLearningLMAccuracy'],
        'language_modeling',
        dataset_uri,
        [],
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

    config = transformers.AutoConfig.from_pretrained('EleutherAI/gpt-neo-125M')
    model = transformers.AutoModelForCausalLM.from_config(config)
    model = HuggingFaceModel(
        model=model,
        tokenizer=None,
        eval_metrics=[InContextLearningLMAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ep', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluators, subset_num_batches=2)
    assert 'metrics/lambada/InContextLearningLMAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/lambada/InContextLearningLMAccuracy'][0][1].item() == 0


@pytest.mark.parametrize('dataset_uri', ['jeopardy_small.jsonl', 'wiki_people_small.jsonl'])
@pytest.mark.parametrize('num_fewshot', [5])
@device('gpu')
def test_lm_task_evaluation_categories(device, dataset_uri, num_fewshot, tiny_gpt2_tokenizer, tmp_path):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    if dataset_uri == 'jeopardy_small.jsonl':
        categories = ['science', 'world_history']
    else:
        categories = ['num_spouses', 'birth_day']
    evaluators = make_evaluators(
        dataset_uri.removesuffix('.jsonl'),
        ['InContextLearningLMAccuracy'],
        'language_modeling',
        dataset_uri,
        categories,
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
    assert len(evaluators) == 3

    config = transformers.AutoConfig.from_pretrained('EleutherAI/gpt-neo-125M')
    model = transformers.AutoModelForCausalLM.from_config(config)
    model = HuggingFaceModel(
        model=model,
        tokenizer=None,
        eval_metrics=[InContextLearningLMAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ep', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluators, subset_num_batches=2)
    assert f"metrics/{dataset_uri.removesuffix('.jsonl')}/science/InContextLearningLMAccuracy" in in_memory_logger.data.keys(
    ) or f"metrics/{dataset_uri.removesuffix('.jsonl')}/num_spouses/InContextLearningLMAccuracy" in in_memory_logger.data.keys(
    )
    assert f"metrics/{dataset_uri.removesuffix('.jsonl')}/world_history/InContextLearningLMAccuracy" in in_memory_logger.data.keys(
    ) or f"metrics/{dataset_uri.removesuffix('.jsonl')}/birth_day/InContextLearningLMAccuracy" in in_memory_logger.data.keys(
    )
    assert 'metrics/jeopardy_small/InContextLearningLMAccuracy' in in_memory_logger.data.keys()

    assert in_memory_logger.data['metrics/jeopardy_small/InContextLearningLMAccuracy'][0][1].item() == 0


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonl', 'hellaswag_small.jsonl'])
@device('gpu')
@world_size(1, 2)
@pytest.mark.parametrize('num_fewshot', [0, 5])
def test_mc_task_evaluation(device, world_size, num_fewshot, dataset_uri, tiny_gpt2_tokenizer, tmp_path,
                            tiny_gpt2_model):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    reproducibility.seed_all(1234)
    tmp_path_to_broadcast = str(os.path.abspath(tmp_path))
    gathered_paths = dist.all_gather_object(tmp_path_to_broadcast)
    evaluators = make_evaluators(
        'mc',
        ['InContextLearningMultipleChoiceAccuracy'],
        'multiple_choice',
        dataset_uri,
        [],
        tokenizer,
        8,
        max_seq_len=1024,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=num_fewshot,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter=': ',
        destination_path=str(Path(gathered_paths[0]) / 'icl.jsonl'),
    )

    config = transformers.AutoConfig.from_pretrained('EleutherAI/gpt-neo-125M')
    model = transformers.AutoModelForCausalLM.from_config(config)
    model = HuggingFaceModel(
        model=tiny_gpt2_model,
        tokenizer=None,
        eval_metrics=[InContextLearningMultipleChoiceAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluators, subset_num_batches=2)
    assert 'metrics/mc/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()


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
    evaluators = make_evaluators(
        'triviaqa',
        ['InContextLearningQAAccuracy'],
        'question_answering',
        dataset_uri,
        [],
        tokenizer,
        8,
        max_seq_len=1024,
        pad_tok_id=tokenizer.eos_token_id,
        num_fewshot=num_fewshot,
        prompt_string='',
        example_delimiter='\n',
        continuation_delimiter=': ',
        destination_path=str(Path(gathered_paths[0]) / 'icl.jsonl'),
    )

    model = HuggingFaceModel(
        model=tiny_gpt2_model,
        tokenizer=tiny_gpt2_tokenizer,
        eval_metrics=[InContextLearningQAAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)

    trainer.eval(eval_dataloader=evaluators, subset_num_batches=2)
    assert 'metrics/triviaqa/InContextLearningQAAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/triviaqa/InContextLearningQAAccuracy'][0][1].item() == 0
