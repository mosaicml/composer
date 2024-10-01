# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Load a checkpoint."""
from typing import Optional

from composer.checkpoint.load import CheckpointLoadOptions, load_checkpoint
from composer.core import Callback, State
from composer.loggers import Logger


class LoadCheckpoint(Callback):
    """Callback that loads a checkpoint after other checkpoints have been loaded.

    Args:
        load_path: The path to the checkpoint to load.
        load_options: A dictionary of options to pass to the checkpoint loading function.
    """

    def __init__(self, load_path: str, load_options: Optional[dict] = None):
        super().__init__()
        self.load_path = load_path
        self.load_options = CheckpointLoadOptions(**(load_options or {}))

    def after_load(self, state: State, logger: Logger) -> None:
        load_checkpoint(
            load_path=self.load_path,
            load_options=self.load_options,
            state=state,
            model_child_path=None,
            optim_child_path=None,
        )
        return super().after_load(state, logger)
