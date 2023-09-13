# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import random

from composer.core import Callback, State
from composer.loggers import Logger


class EvalOutputLogging(Callback):

    def __init__(self, print_only_incorrect=False, subset_sample=-1):
        self.print_only_incorrect = print_only_incorrect
        self.subset_sample = subset_sample

    def prep_response_cache(self, state, cache):
        benchmark = state.dataloader_label
        for metric in state.eval_metrics[benchmark].values():
            if hasattr(metric, 'set_response_cache'):
                metric.set_response_cache(cache)

    def eval_start(self, state: State, logger: Logger) -> None:
        self.prep_response_cache(state, True)

    def eval_end(self, state: State, logger: Logger) -> None:
        assert state.dataloader is not None
        if hasattr(state.dataloader, 'dataset'):
            assert hasattr(state.dataloader, 'dataset')
            if hasattr(state.dataloader.dataset, 'tokenizer'):
                tokenizer = state.dataloader.dataset.tokenizer
                benchmark = state.dataloader_label
                assert benchmark is not None
                assert isinstance(benchmark, str)
                for metric in state.eval_metrics[benchmark].values():
                    if hasattr(metric, 'format_response_cache') and isinstance(metric.format_response_cache, callable):
                        columns, rows = metric.format_response_cache(tokenizer)
                        if columns is not None and rows is not None:
                            if 'correct' not in columns:
                                raise ValueError(f"{type(metric)}'s response cache should have column named `correct`")
                            correct_col = columns.index('correct')
                            if self.print_only_incorrect:
                                rows = [r for r in rows if not r[correct_col]]

                            if self.subset_sample > 0:
                                rows = random.sample(rows, min(len(rows), self.subset_sample))

                            logger.log_table(columns=columns, rows=rows, name=f'icl_outputs/{benchmark}')
        self.prep_response_cache(state, False)
