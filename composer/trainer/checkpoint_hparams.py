# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.trainer.checkpoint import Checkpointer, CheckpointLoader

log = logging.getLogger(__name__)


@dataclass
class CheckpointLoaderHparams(hp.Hparams):
    """Hparams for the :class:`CheckpointLoader`.

    See the documentation for the :class:`CheckpointLoader`.
    """
    filepath: str = hp.required(doc="Path to the serialized state_dict to recover state from.")
    load_weights_only: Optional[bool] = hp.optional(doc="Whether to only load the weights from the model.",
                                                    default=False)
    strict: Optional[bool] = hp.optional(
        doc="Ensure that the set of weights in the checkpoint and model must exactly match.", default=True)

    def validate(self):
        if not self.load_weights_only and self.strict:
            raise ValueError(
                "Strict cannot be used when load_weights_only is true. Restoring a checkpoint from previous state assumes that the checkpoint should perfectly match the model."
            )

    def initialize_object(self) -> CheckpointLoader:
        return CheckpointLoader(checkpoint_filepath=self.filepath,
                                load_weights_only=self.load_weights_only,
                                strict=self.strict)


@dataclass
class CheckpointerHparams(hp.Hparams):
    """Hparams for the :class:`Checkpointer`.

    See the documentation for the :class:`Checkpointer`.
    """
    interval_unit: str = hp.required(
        doc="Unit for the checkpoint save interval -- should be 'ep' for epochs; 'it' for iterations")
    interval: int = hp.required(doc="Interval for checkpointing.")
    folder: str = hp.optional(doc="Folder in which to save checkpoint files. Relative to the run directory, if set."
                              "Defaults to `checkpoints`.",
                              default="checkpoints")

    def validate(self):
        if self.interval < 0:
            raise ValueError("Checkpointing interval must be greater than zero.")
        if self.interval_unit not in ['ep', 'it']:
            raise ValueError("Checkpointing interval unit must be one of 'ep' for epochs, or 'it' for iterations.")

    def initialize_object(self) -> Checkpointer:
        return Checkpointer(checkpoint_interval_unit=self.interval_unit,
                            checkpoint_interval=self.interval,
                            checkpoint_folder=self.folder)
