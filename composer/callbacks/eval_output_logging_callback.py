# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import warnings
from copy import deepcopy
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from composer.core import Callback, State
from composer.loggers import ConsoleLogger, Logger


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are responsible for returning the results of individual datapoints in a dictionary of lists.
    The callback will log the metric name, the depadded and detokenized input, any data stored in state.metric_outputs, and
    any keys from the batch pased into `batch_keys_to_log`. It will do so after every eval batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.warn_batch_is_not_dict = True

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if not isinstance(state.batch, Dict):
            if self.warn_batch_is_not_dict:
                warnings.warn(f'''EvalOutputLogging only supports batchs that are dictionary. \
                    Found batch for type {type(state.batch)}. \
                    Not logging eval outputs.''')
                self.warn_batch_is_not_dict = False
            return

        assert state.outputs is not None
        assert state.metric_outputs is not None
        logging_dict = deepcopy(state.metric_outputs)

        if state.batch['mode'] == 'generate':
            # Outputs are already detokenized
            logging_dict['outputs'] = state.outputs

        input_ids = state.batch['input_ids']
        logged_input = []
        assert state.dataloader is not None
        assert hasattr(state.dataloader, 'dataset')
        assert isinstance(state.dataloader, DataLoader)
        assert state.dataloader.dataset is not None
        assert isinstance(state.dataloader.dataset, Dataset)
        assert hasattr(state.dataloader.dataset, 'pad_tok_id')
        assert state.dataloader.dataset.pad_tok_id is not None
        assert hasattr(state.dataloader.dataset, 'tokenizer')
        # Depad and decode input_ids
        for input_list in input_ids.tolist():
            depadded_input = [tok for tok in input_list if tok != state.dataloader.dataset.pad_tok_id]
            logged_input.append(state.dataloader.dataset.tokenizer.decode(depadded_input))
        logging_dict['input'] = logged_input

        # Get column names
        columns = list(logging_dict.keys())
        # Convert logging_dict from kv pairs of column name and column values to a list of rows
        rows = [list(item) for item in zip(*logging_dict.values())]

        # NOTE: This assumes _any_ tensor logged are tokens to be decoded.
        #       This might not be true if, for example, logits are logged.
        # detokenize data in rows
        rows = [[state.dataloader.dataset.tokenizer.decode(x) if isinstance(x, torch.Tensor) else x
                 for x in row]
                for row in rows]

        assert state.dataloader_label is not None
        step = state.timestamp.batch.value
        name = f'{state.dataloader_label}_step_{step}'
        for dest_logger in logger.destinations:
            if not isinstance(dest_logger, ConsoleLogger):
                dest_logger.log_table(columns, rows, name=name, step=state.timestamp.batch.value)
