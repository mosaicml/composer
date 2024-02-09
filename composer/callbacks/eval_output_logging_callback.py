# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

from copy import deepcopy
from typing import Any, List, Optional

import torch

from composer.core import Callback, State
from composer.loggers import Logger


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are responsible for returning the results of individual datapoints in a dictionary of lists.
    The callback will log the metric name, the depadded and detokenized input, any data stored in state.metric_outputs, and
    any keys from the batch pased into `batch_keys_to_log`. It will do so after every eval batch.
    """

    def __init__(self, batch_keys_to_log: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_keys_to_log = batch_keys_to_log or []

    def eval_after_all(self, state: State) -> None:
        state.metric_outputs = {}

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        assert state.outputs is not None
        assert state.metric_outputs is not None

        logging_dict = deepcopy(state.metric_outputs)
        if state.batch['mode'] == 'generate':
            # Outputs are already detokenized
            logging_dict['outputs'] = state.outputs
        logging_dict['metric_name'] = [state.metric_outputs['metric_name'] for _ in range(0, len(state.outputs))]

        # Depad and decode input_ids
        input_tensor = state.batch['input_ids']
        logged_input = []
        assert state.dataloader is not None
        assert hasattr(state.dataloader, 'dataset')
        assert hasattr(state.dataloader.dataset, 'tokenizer')
        # assert state.dataloader.dataset is not None
        # assert state.dataloader.dataset.tokenizer is not None
        for input_list in input_tensor:
            depadded_input = [tok for tok in input_list if tok != state.dataloader.dataset.pad_tok_id]
            logged_input.append(state.dataloader.dataset.tokenizer.decode(depadded_input))
        logging_dict['input'] = logged_input

        # Log anything from the batch that's specified in the yaml
        for key in self.batch_keys_to_log:
            data_to_log = state.batch[key]
            if isinstance(data_to_log, list):
                logging_dict[key] = state.batch[key]
            else:
                logging_dict[key] = [data_to_log for _ in range(0, len(logging_dict['outputs']))]

        columns = list(logging_dict.keys())
        rows = [list(item) for item in zip(*logging_dict.values())]
        # TODO: This assumes _any_ tensor logged are tokens to be decoded.
        #       This might not be true if, for example, logits are logged.
        # detokenize data in rows
        rows = [[state.dataloader.dataset.tokenizer.decode(x) if isinstance(x, torch.Tensor) else x
                 for x in row]
                for row in rows]

        # TODO:
        # wandb: WARNING Step only supports monotonically increasing values, use define_metric to set a custom x axis. For details see: https://wandb.me/define-metric
        # wandb: WARNING (User provided step: 0 is less than current step: 164. Dropping entry: {'metrics/human_eval/0-shot/InContextLearningCodeEvalAccuracy': 0.0, '_timestamp': 1707370410.1504738}).
        assert state.dataloader_label is not None
        name = state.dataloader_label
        # TODO: How else to chose this?
        for dest_logger in logger.destinations:
            if dest_logger.__class__.__name__ == 'WandBLogger':
                dest_logger.log_table(columns, rows, name=name)
        state.metric_outputs = {}
