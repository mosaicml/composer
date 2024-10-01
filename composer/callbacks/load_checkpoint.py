# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Load a checkpoint."""
import logging
from typing import Optional, Union

from composer.checkpoint.load import CheckpointLoadOptions, load_checkpoint
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
        load_options: Optional[dict] = None,
        event: Union[str, Event] = Event.BEFORE_LOAD,
    ):
        super().__init__()
        self.load_path = load_path
        self.load_options = CheckpointLoadOptions(**(load_options or {}))
        self.event = event if isinstance(event, Event) else Event[event.lower()]

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == self.event:
            self._load(state, logger)
        return super().run_event(event, state, logger)

    def _load(self, state: State, logger: Logger) -> None:
        del logger  # unused

        log.info(f'Loading checkpoint from {self.load_path} at event {self.event}.')

        load_checkpoint(
            load_path=self.load_path,
            load_options=self.load_options,
            state=state,
            model_child_path='',
            optim_child_path='',
        )

        log.info(f'Finished loading checkpoint from {self.load_path} at event {self.event}.')
