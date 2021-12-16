# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import yahp as hp
import yaml

from composer.core.types import StateDict
from composer.trainer.checkpoint import CheckpointLoader
from composer.trainer.devices.device import Device

log = logging.getLogger(__name__)


@dataclass
class CheckpointLoaderHparams(hp.Hparams):
    """Hparams for the :class:`CheckpointLoader`.

    See the documentation for the :class:`CheckpointLoader`.
    """
    checkpoint_filepath: Optional[str] = hp.optional(doc="Path to the serialized state_dict to recover state from.",
                                                     default=None)
    load_weights_only: Optional[bool] = hp.optional(doc="Whether to only load the weights from the model.",
                                                    default=False)
    strict: Optional[bool] = hp.optional(
        doc="Ensure that the set of weights in the checkpoint and model must exactly match.", default=True)

    def initialize_object(self) -> CheckpointLoader:
        print(self.checkpoint_filepath, "filepath")
        if self.checkpoint_filepath:
            return CheckpointLoader(checkpoint_filepath=self.checkpoint_filepath,
                                    load_weights_only=self.load_weights_only,
                                    strict=self.strict)
        return None
