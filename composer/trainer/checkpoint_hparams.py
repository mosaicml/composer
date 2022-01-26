# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import yahp as hp

from composer.utils.object_store import ObjectStoreProviderHparams

if TYPE_CHECKING:
    from composer.trainer.checkpoint import CheckpointLoader, CheckpointSaver

log = logging.getLogger(__name__)


@dataclass
class CheckpointLoaderHparams(hp.Hparams):
    """Hparams for the :class:`CheckpointLoader`.

    See the documentation for the :class:`CheckpointLoader`.
    """
    path: str = hp.required(doc=textwrap.dedent("""The path to an existing checkpoint file
        (if the checkpoint is on the local disk) or the object name for the checkpoint
        (if the checkpoint is in a cloud bucket)."""))
    object_store: Optional[ObjectStoreProviderHparams] = hp.optional(doc=textwrap.dedent("""
        If the checkpoint is in an object store (i.e. AWS S3 or Google Cloud Storage), the parameters for
        connecting to the cloud provider object store. Otherwise, if the checkpoint is a local filepath,
        leave blank."""),
                                                                     default=None)
    load_weights_only: bool = hp.optional(doc="Whether to only load the weights from the model.", default=False)
    strict_model_weights: bool = hp.optional(
        doc="Ensure that the set of weights in the checkpoint and model must exactly match.", default=False)

    chunk_size: int = hp.optional(doc=textwrap.dedent("""Chunk size (in bytes) to use when downloading checkpoints.
        Ignored if the checkpoint is a local file path."""),
                                  default=1_048_576)
    progress_bar: bool = hp.optional(doc=textwrap.dedent("""Whether to show a progress bar when downloading checkpoints.
        Ignored if the checkpoint is a local file path"""),
                                     default=True)

    def validate(self):
        if not self.load_weights_only and self.strict_model_weights:
            raise ValueError(
                "Strict cannot be used when load_weights_only is true. Restoring a checkpoint from previous state assumes that the checkpoint should perfectly match the model."
            )

    def initialize_object(self) -> CheckpointLoader:
        from composer.trainer.checkpoint import CheckpointLoader

        return CheckpointLoader(
            path=self.path,
            object_store_hparams=self.object_store,
            load_weights_only=self.load_weights_only,
            strict_model_weights=self.strict_model_weights,
            chunk_size=self.chunk_size,
            progress_bar=self.progress_bar,
        )


@dataclass
class CheckpointSaverHparams(hp.Hparams):
    """Hparams for the :class:`CheckpointSaver`.

    See the documentation for the :class:`CheckpointSaver`.
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

    def initialize_object(self) -> CheckpointSaver:
        from composer.trainer.checkpoint import CheckpointSaver
        return CheckpointSaver(interval_unit=self.interval_unit, interval=self.interval, save_folder=self.folder)
