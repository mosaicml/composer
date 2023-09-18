# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import hashlib
import os
import random
import shutil
import time
from typing import Callable, Optional

import pandas as pd
from torch.utils.data import DataLoader

from composer.core import Callback, State
from composer.datasets.in_context_learning_evaluation import (InContextLearningCodeEvalDataset,
                                                              InContextLearningLMTaskDataset,
                                                              InContextLearningMultipleChoiceTaskDataset,
                                                              InContextLearningQATaskDataset,
                                                              InContextLearningSchemaTaskDataset)
from composer.loggers import Logger
from composer.utils import maybe_create_object_store_from_uri, parse_uri

ICLDatasetTypes = (InContextLearningLMTaskDataset, InContextLearningQATaskDataset,
                   InContextLearningMultipleChoiceTaskDataset, InContextLearningSchemaTaskDataset,
                   InContextLearningCodeEvalDataset)


def _write(destination_path, src_file):
    obj_store = maybe_create_object_store_from_uri(destination_path)
    _, _, save_path = parse_uri(destination_path)
    if obj_store is not None:
        obj_store.upload_object(object_name=save_path, filename=src_file)
    else:
        shutil.copy(src_file, destination_path)


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are also responsible for providing a method for rendering the cached responses as strings.
    This callback then accesses each eval benchmark during eval_end, retrieves the cached results,
    and renders and and logs them in tabular format.

    If print_only_incorrect=False, correct model outputs will be omitted. If subset_sample > 0, then
    only `subset_sample` of the outputs will be logged.
    """

    def __init__(self,
                 print_only_incorrect: bool = False,
                 subset_sample: int = -1,
                 output_directory: Optional[str] = None):
        self.print_only_incorrect = print_only_incorrect
        self.subset_sample = subset_sample
        self.tables = {}
        self.output_directory = output_directory if output_directory else os.getcwd()
        self.hash = hashlib.sha256()
        self.most_recent_table_paths = None

    def write_tables_to_output_dir(self, state: State):
        # write tmp files
        self.hash.update((str(time.time()) + str(random.randint(0, 1_000_000))).encode('utf-8'))
        tmp_dir = os.getcwd() + '/' + self.hash.hexdigest()
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        table_paths = []
        for table_name in self.tables:
            file_name = f"{table_name.replace('/', '-')}-ba{state.timestamp.batch.value}.tsv"
            with open(f"{tmp_dir}/{file_name}", 'w') as f:
                cols, rows = self.tables[table_name]
                df = pd.DataFrame.from_records(data=rows, columns=cols)
                df.to_csv(f, sep='\t', index=False)
                table_paths.append(file_name)

        # copy/upload tmp files
        for tmp_tbl_path in table_paths:
            _write(destination_path=f"{self.output_directory}/{file_name}", src_file=f"{tmp_dir}/{file_name}")
            os.remove(f"{tmp_dir}/{file_name}")

        # delete tmp files
        os.rmdir(tmp_dir)
        self.most_recent_table_paths = [f"{self.output_directory}/{file_name}" for file_name in tmp_tbl_path]

    def prep_response_cache(self, state, cache):
        benchmark = state.dataloader_label
        for metric in state.eval_metrics[benchmark].values():
            if hasattr(metric, 'set_response_cache'):
                metric.set_response_cache(cache)

    def eval_after_all(self, state: State, logger: Logger) -> None:
        self.write_tables_to_output_dir(state)

    def eval_standalone_end(self, state: State, logger: Logger) -> None:
        self.write_tables_to_output_dir(state)

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
