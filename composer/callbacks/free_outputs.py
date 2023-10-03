# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Free train metrics."""

import torch

from composer.core import Callback, State
from composer.loggers import Logger


class FreeOutputs(Callback):
    """Free train metrics on AFTER_LOSS to reduce peak memory usage if not using train metrics."""

    def after_loss(self, state: State, logger: Logger) -> None:
        state.outputs = torch.Tensor()
