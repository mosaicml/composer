# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log model outputs and expected outputs during ICL evaluation."""

import logging
from typing import Any, List, Optional
from copy import deepcopy

from composer.core import Callback, State
from composer.loggers import Logger

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

        logger.log_table(columns, rows, name=state.dataloader_label)
        state.metric_outputs = None
