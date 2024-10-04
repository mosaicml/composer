# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Load a checkpoint."""
import logging
from typing import Optional, Union

from composer.core import Callback, State
from composer.core.event import Event
from composer.loggers import Logger
from composer.models.huggingface import HuggingFaceModel
from composer.utils.checkpoint import load_checkpoint
from composer.utils.file_helpers import maybe_create_object_store_from_uri, parse_uri

log = logging.getLogger(__name__)


class LoadCheckpoint(Callback):
    """Callback that loads a checkpoint at the specified event.

    Args:
        load_path (str): The path to the checkpoint to load.
        load_options (Optional[dict]): A dictionary of options to pass to the checkpoint loading function.
        event (Union[str, Event]): The event at which to load the checkpoint. Defaults to ``Event.BEFORE_LOAD``.
    """

    def __init__(
        self,
        load_path: str,
        load_weights_only: bool = False,
        strict_model_weights: bool = True,
        ignore_keys: Optional[list[str]] = None,
        event: Union[str, Event] = Event.BEFORE_LOAD,
    ):
        super().__init__()
        self.load_path = load_path
        self.load_object_store = maybe_create_object_store_from_uri(load_path)
        _, _, self.parsed_path = parse_uri(load_path)

        self.load_weights_only = load_weights_only
        self.strict_model_weights = strict_model_weights
        self.ignore_keys = ignore_keys

        self.event = event if isinstance(event, Event) else Event[event.upper()]

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == self.event:
            log.info(f'Loading checkpoint from {self.load_path} at {self.event}.')
            self._load(state, logger)
            log.info(f'Finished loading checkpoint from {self.load_path} at {self.event}.')

        return super().run_event(event, state, logger)

    def _load(self, state: State, logger: Logger) -> None:

        # We need to temporarily disable the `should_save_peft_only` flag on the model
        # so that we can have access to the full model weights for loading.
        model = state.model
        original_should_save_peft_only = False
        if isinstance(model, HuggingFaceModel):
            original_should_save_peft_only = model.should_save_peft_only
            model.should_save_peft_only = False

        load_checkpoint(
            path=self.parsed_path,
            state=state,
            logger=logger,
            object_store=self.load_object_store,
            strict_model_weights=self.strict_model_weights,
            ignore_keys=self.ignore_keys,
            load_weights_only=self.load_weights_only,
        )

        # Restore the original `should_save_peft_only` flag on the model
        if isinstance(model, HuggingFaceModel):
            model.should_save_peft_only = original_should_save_peft_only
