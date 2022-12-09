# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from itertools import product

import pytest
import transformers

from composer.core import Evaluator, Trainer
from composer.datasets.in_context_learning_evaluation import get_perplexity_task_dataloader


@pytest.mark.parametrize('dataset_uri', ['lambada_test.json'])
@pytest.mark.datafiles('local_data/lambada_test.json')
def test_get_perplexity_task_dataloader(dataset_uri):
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    eval_dataloader = get_perplexity_task_dataloader(dataset_uri, tokenizer, 16)
    print(eval_dataloader)

    eval_evaluator = Evaluator(label='myEvaluator', dataloader=eval_dataloader, metric_names=['LAMBADAAccuracy'])

    trainer = Trainer(
        model=model,
        eval_dataloader=eval_evaluator,
        optimizers=optimizer,
        max_duration='1ep',
    )
