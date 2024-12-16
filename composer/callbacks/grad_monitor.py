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

    def __init__(
        self,
    )-> None:
        self.num_microbatches = 0
        self.executed_steps = 0

    def _extract_grads(self, state: State, device: torch.device = torch.device('cpu')) -> None:
        """Extracts gradients of each batch from the model
        A running average of the gradients is stored in the state.

        Args:
            state (State): The state object.
            device (torch.device, optional): The device to store the gradients. Defaults to CPU.
        """
        
        group = list(state.model.parameters())
        grad_list = []
        for p in group:
            if p.grad is not None:
                grad_list.append(p.grad.detach().clone().to(device))
        
        # average the gradients
        prev_grads = state.grads
        if prev_grads:
            aver_grad_list = [(prev_grads[i] * self.executed_steps + grad_list[i]) / (self.executed_steps + 1) for i in range(len(prev_grads))]
        else: # the first batch, no need to average
            aver_grad_list = grad_list
        
        self.executed_steps = self.executed_steps + 1

        if self.executed_steps == state.local_steps: # averaged gradients will be sent to the cloud, so we can reset the counter
            self.executed_steps = 0
        
        state.grads = aver_grad_list
    

    def after_backward(self, state: State, logger: Logger) -> None:
        """Runs on ``Event.AFTER_BACKWARD`` in the function of _train_microbatch.
        """
        assert state.total_num_microbatches is not None, "The total number of microbatch must be set"
        self.num_microbatches = self.num_microbatches + 1
        if self.num_microbatches == state.total_num_microbatches:
            self.num_microbatches = 0
            self._extract_grads(state)
