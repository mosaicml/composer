# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from pathlib import Path

import pytest
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from composer import Evaluator
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
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ': Pour it onto a plate'
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
    assert tokenizer.decode(batch['input_ids'][0][min_idx:max_idx + 1]) == ': Pour it onto a plate'


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

    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])

    model = HuggingFaceModel(
        model=tiny_gpt2_model,
        tokenizer=None,
        eval_metrics=[InContextLearningMultipleChoiceAccuracy()],
        use_logits=True,
    )

    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator, subset_num_batches=2)
    assert 'metrics/lambada/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/lambada/InContextLearningMultipleChoiceAccuracy'][0][1].item() > 0


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
