# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Callback for catching loss NaNs."""

from typing import Sequence

import torch

from composer import Callback, Logger, State

__all__ = ['NaNMonitor']


class NaNMonitor(Callback):
    """Catches NaNs in the loss and raises an error if one is found."""

    def after_loss(self, state: State, logger: Logger):
        """Check if loss is NaN and raise an error if so."""
        if isinstance(state.loss, torch.Tensor):
            if torch.isnan(state.loss).any():
                raise RuntimeError('Train loss contains a NaN.')
        elif isinstance(state.loss, Sequence):
            for loss in state.loss:
                if torch.isnan(loss).any():
                    raise RuntimeError('Train loss contains a NaN.')
        elif isinstance(state.loss, dict):
            for k, v in state.loss.items():
                if torch.isnan(v).any():
                    raise RuntimeError(f'Train loss {k} contains a NaN.')
        else:
            raise TypeError(f'Loss is of type {type(state.loss)}, but should be a tensor or a sequence of tensors')
