# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import hashlib
import os
import random
import shutil
import tempfile
from typing import Callable, Optional

from torch.utils.data import DataLoader

from composer.core import Callback, State
from composer.datasets.in_context_learning_evaluation import (InContextLearningCodeEvalDataset,
                                                              InContextLearningLMTaskDataset,
                                                              InContextLearningMultipleChoiceTaskDataset,
                                                              InContextLearningQATaskDataset,
                                                              InContextLearningSchemaTaskDataset)
from composer.loggers import Logger
from composer.loggers.console_logger import ConsoleLogger
from composer.utils import MissingConditionalImportError, dist, maybe_create_object_store_from_uri, parse_uri

ICLDatasetTypes = (InContextLearningLMTaskDataset, InContextLearningQATaskDataset,
                   InContextLearningMultipleChoiceTaskDataset, InContextLearningSchemaTaskDataset,
                   InContextLearningCodeEvalDataset)


def _write(destination_path, src_file):
    obj_store = maybe_create_object_store_from_uri(destination_path)
    _, _, save_path = parse_uri(destination_path)
    if obj_store is not None:
        obj_store.upload_object(object_name=save_path, filename=src_file)
    else:
        with dist.local_rank_zero_download_and_wait(destination_path):
            if dist.get_local_rank() == 0:
                shutil.copy(src_file, destination_path)


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are also responsible for providing a method for rendering the cached responses as strings.
    This callback then accesses each eval benchmark during eval_end, retrieves the cached results,
    and renders and and logs them in tabular format.

    If subset_sample > 0, then only `subset_sample` of the outputs will be logged.

    output_directory indicates where to write the tsv results, either can be a local directory or a cloud storage directory.
    """

    def __init__(self, subset_sample: int = -1, output_directory: Optional[str] = None):
        self.subset_sample = subset_sample
        self.table = {}
        self.output_directory = output_directory if output_directory else os.getcwd()
        self.hash = hashlib.sha256()
        self.destination_file = None

    # with tempfile.NamedTemporaryFile
    #  tmp_dir =
    def _write_tables_to_output_dir(self, state: State):
        try:
            import pandas as pd
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='pandas',
                                                conda_package='pandas',
                                                conda_channel='conda-forge') from e
        # write tmp files

        full_df = pd.DataFrame()
        upload_file_name = f'eval-outputs-ba{state.timestamp.batch.value}.tsv'

        for benchmark in self.table:
            cols, rows = self.table[benchmark]
            rows = [[e.encode('unicode_escape') if isinstance(e, str) else e for e in row] for row in rows]
            df = pd.DataFrame.from_records(data=rows, columns=cols)
            df['benchmark'] = benchmark
            full_df = pd.concat([full_df, df], ignore_index=True)

        tmp_file = ''
        with tempfile.NamedTemporaryFile('wb') as f:
            full_df.to_csv(f, sep='\t', index=False)
            tmp_file = f.name

        # copy/upload tmp files
        _write(destination_path=f'{self.output_directory}/{upload_file_name}', src_file=tmp_file)
        self.destination_file = f'{self.output_directory}/{upload_file_name}'

    def _prep_response_cache(self, state, cache):
        benchmark = state.dataloader_label
        for metric in state.eval_metrics[benchmark].values():
            if hasattr(metric, 'reset_response_cache'):
                metric.reset_response_cache(cache)

    def eval_start(self, state: State, logger: Logger) -> None:
        # eval start runs before each benchmark's evaluator (either in training or eval)
        self._prep_response_cache(state, True)

    def eval_after_all(self, state: State, logger: Logger) -> None:
        # eval after all runs after all evaluators have completed during eval within training
        #  (either in training or eval)
        self._write_tables_to_output_dir(state)
        self.table = {}

    def eval_standalone_end(self, state: State, logger: Logger) -> None:
        # eval standalone end runs after all evaluators have completed during a direct call to trainer.eval()
        self._write_tables_to_output_dir(state)
        self.table = {}

    def eval_end(self, state: State, logger: Logger) -> None:
        # eval start runs after each benchmark's evaluator
        # during each eval, only a single dataloader/benchmark will be active
        assert state.dataloader is not None
        assert isinstance(state.dataloader, DataLoader)
        if hasattr(state.dataloader, 'dataset') and isinstance(state.dataloader.dataset, ICLDatasetTypes):
            assert isinstance(state.dataloader.dataset, ICLDatasetTypes)
            if hasattr(state.dataloader.dataset, 'tokenizer'):
                tokenizer = state.dataloader.dataset.tokenizer
                benchmark = state.dataloader_label
                assert benchmark is not None
                assert isinstance(benchmark, str)
                for metric_name, metric in state.eval_metrics[benchmark].items():
                    if hasattr(metric, 'format_response_cache'):
                        assert isinstance(metric.format_response_cache, Callable)
                        format_response_cache: Callable = metric.format_response_cache
                        columns, rows = format_response_cache(tokenizer)

                        if columns is not None and rows is not None:
                            if self.subset_sample > 0:
                                rows = random.sample(rows, min(len(rows), self.subset_sample))
                            for destination in logger.destinations:
                                if not isinstance(destination, ConsoleLogger):
                                    # don't log to console because it will pollute the console too much
                                    destination.log_table(columns, rows, f'icl_outputs/{benchmark}/{metric_name}')

                            self.table[f'{benchmark}_{metric_name}'] = (columns, rows)
        self._prep_response_cache(state, False)
