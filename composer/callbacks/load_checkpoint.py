# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Load a checkpoint."""
import logging
from typing import List, Optional, Union

from composer.utils.checkpoint import load_checkpoint
from composer.core import Callback, State
from composer.core.event import Event
from composer.loggers import Logger

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
        ignore_keys: Optional[List[str]] = None,
        event: Union[str, Event] = Event.AFTER_LOAD,
    ):
        super().__init__()
        self.load_path = load_path
        self.load_weights_only = load_weights_only
        self.strict_model_weights = strict_model_weights
        self.ignore_keys = ignore_keys

        self.event = event if isinstance(event, Event) else Event[event.lower()]

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == self.event:
            log.info(f'Loading checkpoint from {self.load_path} at event {self.event}.')
            self._load(state, logger)
            log.info(f'Finished loading checkpoint from {self.load_path} at event {self.event}.')

        return super().run_event(event, state, logger)

    def _load(self, state: State, logger: Logger) -> None:
        print('state state dict', state.state_dict()['model'].keys())
        load_checkpoint(
            path=self.load_path,
            state=state,
            logger=logger,
            strict_model_weights=self.strict_model_weights,
            ignore_keys=self.ignore_keys,
            load_weights_only=self.load_weights_only,
        )

