# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from torch.utils.data import DataLoader

from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import (_get_fewshot_sample_idxs, _make_padded_input,
                                                              get_icl_task_dataloader)
from composer.loggers import InMemoryLogger
from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer
from tests.common import device


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


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonl'])
def test_lm_task_dataloader(dataset_uri, tiny_gpt2_tokenizer):
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
                                 num_fewshot=1,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter='')

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


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonl'])
def test_mc_task_dataloader(dataset_uri, tiny_gpt2_tokenizer):
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
                                 continuation_delimiter=': ')

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
def test_lm_task_evaluation(device, dataset_uri, num_fewshot, tiny_gpt2_tokenizer):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    dl = get_icl_task_dataloader('language_modeling',
                                 dataset_uri,
                                 tokenizer,
                                 2,
                                 max_seq_len=2048,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=num_fewshot,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter='')
    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningLMAccuracy'])
    model = create_gpt2(use_pretrained=False, pretrained_model_name='EleutherAI/gpt-neo-125M')
    model.add_eval_metrics(evaluator)
    trainer = Trainer(model=model, max_duration='1ep', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator, subset_num_batches=2)
    assert 'metrics/lambada/InContextLearningLMAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/lambada/InContextLearningLMAccuracy'][0][1].item() == 0


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonl', 'hellaswag_small.jsonl'])
@device('gpu')
@pytest.mark.parametrize('num_fewshot', [0, 5])
def test_mc_task_evaluation(device, num_fewshot, dataset_uri, tiny_gpt2_tokenizer):
    pytest.importorskip('datasets')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    dl = get_icl_task_dataloader('multiple_choice',
                                 dataset_uri,
                                 tokenizer,
                                 8,
                                 max_seq_len=2048,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=num_fewshot,
                                 prompt_string='',
                                 example_delimiter='\n',
                                 continuation_delimiter=': ')
    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])
    model = create_gpt2(use_pretrained=False, pretrained_model_name='EleutherAI/gpt-neo-125M')
    model.add_eval_metrics(evaluator)
    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator, subset_num_batches=2)
    assert 'metrics/lambada/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/lambada/InContextLearningMultipleChoiceAccuracy'][0][1].item() > 0
