# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn

from composer.core.types import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)


class SWA(Algorithm):
    """Apply Stochastic Weight Averaging (`Izmailov et al. <https://arxiv.org/abs/1803.05407>`_)

    Stochastic Weight Averaging (SWA) averages model weights sampled at
    different times near the end of training. This leads to better
    generalization than just using the final trained weights.

    Because this algorithm needs to maintain both the current value of the
    weights and the average of all of the sampled weights, it doubles the
    model's memory consumption. Note that this does not mean that the total
    memory required doubles, however, since stored activations and the
    optimizer state are not doubled.

    Args:
        swa_start: fraction of training completed before stochastic weight averaging is applied
        swa_lr: the final learning rate used for weight averaging

    Note that 'anneal_epochs' is not used in the current implementation
    """

    def __init__(self, swa_start: float = 0.8, anneal_epochs: int = 10, swa_lr: Optional[float] = None):
        self.swa_start = swa_start
        self.anneal_epochs = anneal_epochs
        self.swa_lr = swa_lr
        self.swa_model: Optional[torch.nn.Module] = None

        assert 0 < swa_start < 1, "swa_start must be between 0 and 1."
        assert anneal_epochs > 0, "anneal_epochs must be great than 0."

        self.swa_scheduler = None
        self.swa_model = None

    def match(self, event: Event, state: State) -> bool:
        """Run on EPOCH_END if training duration is greater than `swa_start`

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        should_start_swa = float(state.get_elapsed_duration()) >= self.swa_start
        return event == Event.EPOCH_END and should_start_swa

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Apply SWA to weights towards the end of training.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """

        if self.swa_scheduler is None:

            if self.swa_lr is None:
                if len(state.schedulers) != 1:
                    raise RuntimeError("SWA supports only one scheduler")
                scheduler = state.schedulers[0]
                scheduler.get_last_lr()
                last_lr = scheduler.get_last_lr()
                if len(last_lr) != 1:
                    raise RuntimeError("SWA supports only one LR")
                log.info(f'Setting SWA LR to {last_lr}')
                self.swa_lr = last_lr[0]

            if len(state.optimizers) != 1:
                raise RuntimeError("SWA supports one and only one optimizer")

            self.swa_scheduler = SWALR(
                state.optimizers[0],
                swa_lr=self.swa_lr,
                anneal_epochs=self.anneal_epochs,
                anneal_strategy='cos',
            )

        if self.swa_model is None:
            self.swa_model = AveragedModel(state.model)

        self.swa_model.update_parameters(state.model)  # type: ignore

        if self.swa_scheduler is None:
            raise ValueError('SWA LR scheduler was not set.')
        self.swa_scheduler.step()

        ## end of training
        if float(state.get_elapsed_duration()) >= 1.0:
            device = next(self.swa_model.parameters()).device
            # TODO(laura) this does not apply the batch split fn. This may result in cuda OOM
            update_bn(state.train_dataloader, model=self.swa_model, device=device)
            state.model = self.swa_model
            log.info('Updated BN and set model to the averaged model')
