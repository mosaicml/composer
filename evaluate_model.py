
import os
from itertools import product
from composer.loggers import InMemoryLogger

import pytest
import transformers

from composer.core import Evaluator
from composer.trainer import Trainer
from composer.datasets.in_context_learning_evaluation import get_lm_task_dataloader
from composer.models.gpt2 import create_gpt2
from composer.optim import DecoupledSGDW



if __name__ == "__main__":
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    dataset_uri = 'tests/datasets/local_data/lambada_test_small.json'
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    dl = get_lm_task_dataloader(
        dataset_uri,
        tokenizer,
        16,
        max_seq_len=2048,
        eos_tok_id=tokenizer.eos_token_id
    )
    evaluator = Evaluator(
        label='lambada',
        dataloader=dl,
        metric_names=['InContextLearningLMAccuracy']
    )
    model = create_gpt2(
        use_pretrained=True,
        pretrained_model_name='EleutherAI/gpt-neo-1.3B'
    )
    trainer = Trainer(
        model=model,
        max_duration='1ep',
        loggers=in_memory_logger
    )
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/lambada/InContextLearningLMAccuracy' in in_memory_logger.data.keys()
    print(in_memory_logger.data['metrics/lambada/InContextLearningLMAccuracy'])
