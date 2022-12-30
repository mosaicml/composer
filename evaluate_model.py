import os

import pytest
import transformers

from composer.core import Evaluator
from composer.datasets.in_context_learning_language_modeling_evaluation import get_lm_task_dataloader
from composer.datasets.in_context_learning_multiple_choice_evaluation import get_mc_task_dataloader

from composer.loggers import InMemoryLogger
from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer

if __name__ == '__main__':
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    # dataset_uri = 's3://mosaicml-internal-dataset-lambda/lambada/lambada_test.json'
    dataset_uri = 's3://mosaicml-internal-dataset-hellaswag/hellaswag.jsonz'

    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    # dl = get_lm_task_dataloader(dataset_uri,
    #                             tokenizer,
    #                             2,
    #                             max_seq_len=2048,
    #                             eos_tok_id=tokenizer.eos_token_id,
    #                             num_fewshot=1,
    #                             preamble_string='',
    #                             example_delimiter='\n',
    #                             continuation_delimiter='')
    dl = get_mc_task_dataloader(dataset_uri,
                                tokenizer,
                                batch_size=16,
                                max_seq_len=2048,
                                eos_tok_id=tokenizer.eos_token_id,
                                num_fewshot=5,
                                preamble_string='',
                                example_delimiter='\n',
                                continuation_delimiter=': ')

    evaluator = Evaluator(label='hellaswag', dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])
    model = create_gpt2(use_pretrained=True, pretrained_model_name='EleutherAI/gpt-neo-1.3B')
    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/hellaswag/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    print(in_memory_logger.data['metrics/hellaswag/InContextLearningMultipleChoiceAccuracy'][0][1].item())
