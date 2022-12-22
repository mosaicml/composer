import os

import pytest
import transformers

from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import get_lm_task_dataloader
from composer.loggers import InMemoryLogger
from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer

if __name__ == '__main__':
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    dataset_uri = 's3://mosaicml-internal-dataset-lambda/lambada/lambada_test.json'
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    dl = get_lm_task_dataloader(dataset_uri,
                                tokenizer,
                                2,
                                max_seq_len=2048,
                                eos_tok_id=tokenizer.eos_token_id,
                                num_fewshot=1,
                                preamble_string='',
                                example_delimiter='\n',
                                continuation_delimiter='')

    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningLMAccuracy'])
    model = create_gpt2(use_pretrained=True, pretrained_model_name='EleutherAI/gpt-neo-1.3B')
    trainer = Trainer(model=model, max_duration='1ep', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/lambada/InContextLearningLMAccuracy' in in_memory_logger.data.keys()
    print(in_memory_logger.data['metrics/lambada/InContextLearningLMAccuracy'][0][1].item())
