# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during DDP training."""

import warnings
from typing import Union
import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['GradMonitor']


class GradMonitor(Callback):
    """extracts gradients from the self.state.model during training.

    The extracted gradients are stored in the self.state.grads attribute, in the form of a list of tensors.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import GradMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[GradMonitor()],
            ... )

    """

    def __init__(self,):
        pass

    def batch_end(self, state: State, logger: Logger):
        """Called on the :attr:`.Event.BATCH_END` event.
        """
        group = list(state.model.parameters())
        grad_list = []
        for p in group:
            if p.grad is not None:
                grad_list.append(p.grad)
        state.grads = grad_list
