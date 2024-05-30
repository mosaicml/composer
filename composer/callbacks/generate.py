# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations from a set of prompts."""

import logging
import time
from typing import Any, Optional, Union, cast

from composer.core import Callback, Event, State, Time, get_precision_context
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import create_interval_scheduler, dist
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)


class Generate(Callback):
    """Periodically log generations from a set of prompts.

    Args:
        prompts (list[str]): The list of prompts you would like to produce generations for
        interval (Union[str, int, :class:`.Time`]): The interval describing how often checkpoints should be
            saved. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        batch_size (Optional[int]): Size of a prompt batch for generation. If None, defaults to the number of prompts.
        kwargs: All kwargs will be passed along to the call to generate. This is for things like `do_sample`, `top_p`, etc
    """

    def __init__(
        self,
        prompts: list[str],
        interval: Union[str, int, Time],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='nlp',
                conda_package='transformers',
                conda_channel='conda-forge',
            ) from e
        del transformers
        self.prompts = prompts
        self.generate_kwargs = kwargs
        self.batch_size = batch_size if batch_size is not None else len(prompts)
        self.check_interval = create_interval_scheduler(interval, include_end_of_training=True)
        self.last_generate_batch: Optional[Time] = None

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if state.get_elapsed_duration(
        ) is not None and self.check_interval(state, event) and self.last_generate_batch != state.timestamp.batch:
            start = time.time()
            self.generate(state, logger)
            diff = time.time() - start
            log.info(f'Generate callback ran in {diff} seconds for {len(self.prompts)} prompts')

    def generate(self, state: State, logger: Logger):
        self.last_generate_batch = state.timestamp.batch

        model = state.model.module if state.is_model_ddp else state.model
        if not isinstance(model, HuggingFaceModel):  # TODO: Extend to support any models that have a generate method.
            raise ValueError(f'Expected HuggingFaceModel, but got {model.__class__.__name__}')

        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            raise ValueError(
                f'Model {model.__class__.__name__} does not have a tokenizer which is required for generation.',
            )
        tokenizer = model.tokenizer

        from transformers import PreTrainedTokenizerBase
        tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

        # Set to evaluation mode and stash the original mode.
        original_mode = model.training
        model.eval()
        device = state.device

        # Stash the original value of padding_side because generation requires left padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_input = tokenizer(self.prompts, return_tensors='pt', padding=True)

        all_input_ids = tokenized_input['input_ids']
        all_attn_masks = tokenized_input['attention_mask']

        output_token_ids = []
        # dummy forward call needed for FSDP to work consistently
        model.dummy_forward_called = False

        n_prompts = len(self.prompts)
        for start in range(0, n_prompts, self.batch_size):
            end = min(start + self.batch_size, n_prompts)
            input_ids = all_input_ids[start:end]  # pyright: ignore[reportGeneralTypeIssues]
            attn_mask = all_attn_masks[start:end]  # pyright: ignore[reportGeneralTypeIssues]

            # Move batch to device.
            input_ids = device.tensor_to_device(input_ids)
            attn_mask = device.tensor_to_device(attn_mask)
            with get_precision_context(state.precision):
                output_token_ids.extend(
                    model.generate(  # type: ignore
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        synced_gpus=dist.get_world_size() > 1,
                        **self.generate_kwargs,
                    ),
                )

        if dist.get_global_rank() == 0:
            # Process prompts and outputs into a table.
            rows = []
            input_tokens_len = all_input_ids.shape[1]  # pyright: ignore[reportGeneralTypeIssues]
            for i, prompt in enumerate(self.prompts):
                output_tokens = output_token_ids[i][input_tokens_len:]
                output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                rows.append([prompt, output_text])

            logger.log_table(columns=['prompt', 'generation'], rows=rows, name='generations')

        tokenizer.padding_side = original_padding_side
        model.train(mode=original_mode)
