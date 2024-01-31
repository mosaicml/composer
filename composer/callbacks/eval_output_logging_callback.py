# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import logging

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

log = logging.getLogger(__name__)


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are also responsible for providing a method for rendering the cached responses as strings.
    This callback then accesses each eval benchmark during eval_end, retrieves the cached results,
    and renders and and logs them in tabular format.

    If subset_sample > 0, then only `subset_sample` of the outputs will be logged.

    output_directory indicates where to write the tsv results, either can be a local directory or a cloud storage directory.
    """

    def eval_after_all(self, state: State) -> None:
        state.metric_outputs = None

    def eval_batch_end(self, state: State, metric_name: str) -> None:
        assert state.outputs is not None
        assert state.metric_outputs is not None

        columns = list(state.eval_outputs.keys())
        rows = [list(item) for item in zip(*state.outputs.values())]

        log.log_table(columns, rows, name=metric)
