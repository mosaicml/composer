# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import random
from typing import Callable

from torch.utils.data import DataLoader

from composer.core import Callback, State
from composer.datasets.in_context_learning_evaluation import (InContextLearningCodeEvalDataset,
                                                              InContextLearningLMTaskDataset,
                                                              InContextLearningMultipleChoiceTaskDataset,
                                                              InContextLearningQATaskDataset,
                                                              InContextLearningSchemaTaskDataset)
from composer.loggers import Logger

ICLDatasetTypes = (InContextLearningLMTaskDataset, InContextLearningQATaskDataset,
                   InContextLearningMultipleChoiceTaskDataset, InContextLearningSchemaTaskDataset,
                   InContextLearningCodeEvalDataset)


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are also responsible for providing a method for rendering the cached responses as strings.
    This callback then accesses each eval benchmark during eval_end, retrieves the cached results,
    and renders and and logs them in tabular format.

    If print_only_incorrect=False, correct model outputs will be omitted. If subset_sample > 0, then
    only `subset_sample` of the outputs will be logged.
    """

    def __init__(self, print_only_incorrect=False, subset_sample=-1):
        self.print_only_incorrect = print_only_incorrect
        self.subset_sample = subset_sample
        self.tables = {}

    def prep_response_cache(self, state, cache):
        benchmark = state.dataloader_label
        for metric in state.eval_metrics[benchmark].values():
            if hasattr(metric, 'set_response_cache'):
                metric.set_response_cache(cache)

    def eval_start(self, state: State, logger: Logger) -> None:
        self.prep_response_cache(state, True)

    def eval_end(self, state: State, logger: Logger) -> None:
        
        assert state.dataloader is not None
        assert isinstance(state.dataloader, DataLoader)
        if hasattr(state.dataloader, 'dataset') and isinstance(state.dataloader.dataset, ICLDatasetTypes):
            assert isinstance(state.dataloader.dataset, ICLDatasetTypes)
            if hasattr(state.dataloader.dataset, 'tokenizer'):
                tokenizer = state.dataloader.dataset.tokenizer
                benchmark = state.dataloader_label
                assert benchmark is not None
                assert isinstance(benchmark, str)
                for metric in state.eval_metrics[benchmark].values():
                    if hasattr(metric, 'format_response_cache'):
                        assert isinstance(metric.format_response_cache, Callable)
                        format_response_cache: Callable = metric.format_response_cache
                        columns, rows = format_response_cache(tokenizer)
                        if columns is not None and rows is not None:
                            if 'correct' not in columns:
                                raise ValueError(f"{type(metric)}'s response cache should have column named `correct`")
                            correct_col = columns.index('correct')
                            if self.print_only_incorrect:
                                rows = [r for r in rows if not r[correct_col]]

                            if self.subset_sample > 0:
                                rows = random.sample(rows, min(len(rows), self.subset_sample))

                            logger.log_table(columns=columns, rows=rows, name=f'icl_outputs/{benchmark}')
                            self.tables[f'icl_outputs/{benchmark}'] = (columns, rows)
        self.prep_response_cache(state, False)
        return self.tables