# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

from composer.core import Callback, State
from composer.loggers import Logger
from typing import List, Optional
import transformers
import torch
from lm_eval import models as lm_eval_models
from lm_eval import tasks as lm_eval_tasks
import lm_eval
from lm_eval.evaluator import evaluate
import inspect

__all__ = ['EleutherEvalHarness']


def evaluate_model_on_tasks(model: lm_eval.base.LM, tasks: List[str], num_fewshots: List[int], subsample_size: Optional[int]) -> dict:
    '''
    Returns the sum of two decimal numbers in binary digits.

    Parameters:
        model (lm_eval.base.LM): An lm_eval LM model.
        tasks (List[str]): List of task names as defined in lm_eval.tasks.TASK_REGISTRY
        num_fewshots (List[int]): Number of few shots to used for in-context learning for each task. Runs each task
            separately for each number in this list.
        partial_eval_mode (bool): To quickly sanity check model performance, run eval on only the first
            `PARTIAL_EVAL_SAMPLE_SIZE` examples in the task's test set.

    Returns:
        results (dict): Results of the task including ppl, acc, acc_norm (if multiple choice),
            task name, and few shot samples used
    '''
    task_dict = lm_eval_tasks.get_task_dict(tasks)
    results = {}
    for num in num_fewshots:
        eval_res = evaluate(
            lm=model,
            task_dict=task_dict,
            num_fewshot=num,
            limit=subsample_size
        )['results']

        for task_name,task_res in eval_res.items():
            for metric_name in task_res:
                if metric_name in ['ppl', 'acc', 'acc_norm']:
                    results[f"eval/{task_name}/{num}-shot/{metric_name}"] = task_res[metric_name]
       
    return results

class EleutherEvalHarness(Callback):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, task_list: List[str], num_fewshot: List[int], subsample_size: Optional[int] = None):
        self.task_list = task_list
        self.num_fewshot = num_fewshot
        self.subsample_size = subsample_size
        self.tokenizer = tokenizer

    def eval_end(self, state: State, logger: Logger):

        model = state.model
        if inspect.getfullargspec(model.forward).args == ['self', 'batch']:
            model = model.model
        assert "input_ids" in inspect.getfullargspec(model.forward).args

        model_components = {"model": model, "tokenizer": self.tokenizer, "precision": state.precision, "batch_size": 4, "device": 'cuda' if torch.cuda.is_available() else 'cpu'}
        
        wrapped_lm_eval_model = lm_eval_models.get_model("composer_llm").create_from_arg_string(
            "", model_components
        )
        results = evaluate_model_on_tasks(
            wrapped_lm_eval_model,
            self.task_list,
            self.num_fewshot,
            self.subsample_size
        )
        logger.log_metrics(results)

