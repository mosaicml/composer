# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from torch.utils.data import DataLoader

from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import get_lm_task_dataloader
from composer.loggers import InMemoryLogger
from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer
from tests.common import device, world_size


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonz'])
def test_lm_task_dataloader(dataset_uri, tiny_gpt2_tokenizer):
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = tiny_gpt2_tokenizer
    dataset_uri = f'{local_data}/{dataset_uri}'
    dl = get_lm_task_dataloader(dataset_uri, tokenizer, 2, max_seq_len=2048, eos_tok_id=tokenizer.eos_token_id)

    assert isinstance(dl.dataloader, DataLoader)  # pyright
    assert 'input_ids' in next(dl.dataloader._get_iterator())
    assert 'attention_mask' in next(dl.dataloader._get_iterator())
    assert 'continuation_indices' in next(dl.dataloader._get_iterator())
    assert 'labels' in next(dl.dataloader._get_iterator())
    assert 'mode' in next(dl.dataloader._get_iterator())


@pytest.mark.parametrize('dataset_uri', ['lambada_small.jsonz'])
@world_size(1, 2)
@device('cpu', 'gpu')
def test_lm_task_evaluation(device, world_size, dataset_uri, tiny_gpt2_tokenizer):
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    dl = get_lm_task_dataloader(dataset_uri, tokenizer, 2, max_seq_len=2048, eos_tok_id=tokenizer.eos_token_id)
    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningLMAccuracy'])
    model = create_gpt2(use_pretrained=False, pretrained_model_name='EleutherAI/gpt-neo-125M')
    model.add_eval_metrics(evaluator)
    trainer = Trainer(model=model, max_duration='1ep', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/lambada/InContextLearningLMAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/lambada/InContextLearningLMAccuracy'][0][1].item() == 0
