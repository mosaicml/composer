# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import torch
from copy import deepcopy
from typing import Any, List, Optional

from composer.core import Callback, State
from composer.loggers import Logger


class EvalOutputLogging(Callback):
    """Logs eval outputs for each sample of each ICL evaluation dataset.

    ICL metrics are required to support caching the model's responses including information on whether model was correct.
    Metrics are also responsible for providing a method for rendering the cached responses as strings.
    This callback then accesses each eval benchmark during eval_end, retrieves the cached results,
    and renders and and logs them in tabular format.

    If subset_sample > 0, then only `subset_sample` of the outputs will be logged.

    output_directory indicates where to write the tsv results, either can be a local directory or a cloud storage directory.
    """

    def __init__(self, batch_keys_to_log: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_keys_to_log = batch_keys_to_log or []

    def eval_after_all(self, state: State) -> None:
        state.metric_outputs = None

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        assert state.outputs is not None
        assert state.metric_outputs is not None

        logging_dict = deepcopy(state.metric_outputs)
        if state.batch['mode'] == 'generate':
            # Outputs are already detokenized
            logging_dict['outputs'] = state.outputs
        logging_dict['metric_name'] = [state.metric_outputs['metric_name'] for _ in range(0, len(state.outputs))]

        # Decode and depad input_ids
        input_tensor = state.batch['input_ids']
        logged_input = []
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
        # detokenize data in rows
        # TODO: This assumes _any_ tensor logged are tokens to be decoded.
        #       This might not be true if, for example, logits are logged.
        rows = [[state.dataloader.dataset.tokenizer.decode(x) if isinstance(x, torch.Tensor) else x for x in row] for row in rows]

        # TODO:
        # wandb: WARNING Step only supports monotonically increasing values, use define_metric to set a custom x axis. For details see: https://wandb.me/define-metric
        # wandb: WARNING (User provided step: 0 is less than current step: 164. Dropping entry: {'metrics/human_eval/0-shot/InContextLearningCodeEvalAccuracy': 0.0, '_timestamp': 1707370410.1504738}).

        # TODO: How else to chose this?
        for dest_logger in logger.destinations:
            if dest_logger.__class__.__name__ == 'WandBLogger':
                dest_logger.log_table(columns, rows, name=state.dataloader_label)
        state.metric_outputs = None
