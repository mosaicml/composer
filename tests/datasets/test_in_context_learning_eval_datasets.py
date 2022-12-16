# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import transformers

from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import get_lm_task_dataloader
from composer.loggers import InMemoryLogger
from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer


@pytest.mark.parametrize('dataset_uri', ['lambada_test_small.jsonz'])
def test_get_lm_task_dataloader(dataset_uri):
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    dl = get_lm_task_dataloader(dataset_uri, tokenizer, 16, max_seq_len=2048, eos_tok_id=tokenizer.eos_token_id)
    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningLMAccuracy'])
    model = create_gpt2(use_pretrained=True, pretrained_model_name='EleutherAI/gpt-neo-125M')
    trainer = Trainer(model=model, max_duration='1ep', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/lambada/InContextLearningLMAccuracy' in in_memory_logger.data.keys()
