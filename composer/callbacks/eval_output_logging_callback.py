# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import warnings
from copy import deepcopy
from typing import Any, Sequence, Union

import torch

from composer.core import Callback, State
from composer.loggers import ConsoleLogger, Logger
from composer.utils import VersionedDeprecationWarning, dist


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are responsible for returning the results of individual data points in a dictionary of lists.
    The callback will log the metric name, the depadded and detokenized input, any data stored in state.metric_outputs, and
    any keys from the batch passed into `batch_keys_to_log`. It will do so after every eval batch.
    """

    def __init__(self, log_tokens=False, *args, **kwargs):
        warnings.warn(
            VersionedDeprecationWarning(
                '`InContextLearningMetric` and it\'s subclasses have been deprecated and ' +
                'migrated to MosaicML\'s llm-foundry repo under the llmfoundry.eval.datasets.in_context_learning module: '
                + 'https://github.com/mosaicml/llm-foundry/blob/main/scripts/eval/README.md.' +
                'As EvalOutputLogging only works for ICL metrics, it has been deprecated and ' +
                'will be migrated as well.',
                remove_version='0.24.0',
            ),
        )
        super().__init__(self, *args, **kwargs)
        self.log_tokens = log_tokens
        self.columns = None
        self.name = None
        self.rows = []

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if not isinstance(state.batch, dict):
            warnings.warn(
                f'''EvalOutputLogging only supports batches that are dictionary. \
                Found batch for type {type(state.batch)}. \
                Not logging eval outputs.''',
            )
            return

        assert state.outputs is not None
        assert state.metric_outputs is not None
        logging_dict: dict[str, Union[list[Any], torch.Tensor, Sequence[torch.Tensor]]] = deepcopy(state.metric_outputs)

        # If batch mode is not generate, outputs will be logits
        if state.batch['mode'] == 'generate':
            # Outputs are already detokenized
            logging_dict['outputs'] = state.outputs

        input_ids = state.batch['input_ids']
        logged_input = []
        assert state.dataloader is not None

        # Depad and decode input_ids
        for input_list in input_ids.tolist():
            dataset = state.dataloader.dataset  # pyright: ignore[reportGeneralTypeIssues]
            depadded_input = [tok for tok in input_list if tok != dataset.pad_tok_id]
            logged_input.append(dataset.tokenizer.decode(depadded_input))
        logging_dict['input'] = logged_input

        # Log token indices if toggled
        if self.log_tokens:
            logging_dict['input_tokens'] = input_ids.tolist()
            if not state.batch['mode'] == 'generate':
                if isinstance(state.outputs, torch.Tensor):  # pyright
                    logging_dict['label_tokens'] = state.outputs.tolist()

        # Add run_name as a column
        run_name_list = [state.run_name for _ in range(0, len(logging_dict['input']))]
        logging_dict['run_name'] = run_name_list

        # NOTE: This assumes _any_ tensor logged are tokens to be decoded.
        #       This might not be true if, for example, logits are logged.

        # Detokenize data in rows
        for key, value in logging_dict.items():
            # All types in list are the same
            if isinstance(value[0], torch.Tensor):
                logging_dict[key] = [
                    state.dataloader.dataset.tokenizer.decode(t)  # pyright: ignore[reportGeneralTypeIssues]
                    for t in value
                ]
            elif isinstance(value[0], list):
                if isinstance(value[0][0], torch.Tensor):
                    tokenizer = state.dataloader.dataset.tokenizer  # pyright: ignore[reportGeneralTypeIssues]
                    logging_dict[key] = [[tokenizer.decode(choice) for choice in t] for t in value]

        # Convert logging_dict from kv pairs of column name and column values to a list of rows
        # Example:
        # logging_dict = {"a": ["1a", "2a"], "b": ["1b", "2b"]}
        # will become
        # columns = {"a", "b"}, rows = [["1a", "1b"], ["2a", "2b"]]
        columns = list(logging_dict.keys())
        rows = [list(item) for item in zip(*logging_dict.values())]

        assert state.dataloader_label is not None
        if not self.name:
            # If only running eval, step will be 0
            # If running training, step will be current training step
            step = state.timestamp.batch.value
            self.name = f'{state.dataloader_label}_step_{step}'
            self.columns = columns
        self.rows.extend(rows)

    def eval_end(self, state: State, logger: Logger) -> None:
        list_of_rows = dist.all_gather_object(self.rows)
        rows = [row for rows in list_of_rows for row in rows]
        for dest_logger in logger.destinations:
            if not isinstance(dest_logger, ConsoleLogger):
                dest_logger.log_table(self.columns, rows, name=self.name, step=state.timestamp.batch.value)

        self.rows = []
        self.name = None
        self.columns = None
