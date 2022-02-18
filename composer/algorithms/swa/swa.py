# Copyright 2021 MosaicML. All Rights Reserved.

"""Core code for Stochastic Weight Averaging."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn

from composer.core.types import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)

__all__ = ['SWA']


class SWA(Algorithm):
    """Implements Stochastic Weight Averaging (`Izmailov et al, 2018 <https://arxiv.org/abs/1803.05407>`_).

    Stochastic Weight Averaging (SWA) averages model weights sampled at
    different times near the end of training. This leads to better
    generalization than just using the final trained weights.

    Because this algorithm needs to maintain both the current value of the
    weights and the average of all of the sampled weights, it doubles the
    model's memory consumption. Note that this does not mean that the total
    memory required doubles, however, since stored activations and the
    optimizer state are not doubled.

    This algorithm runs on :attr:`~composer.core.event.Event.EPOCH_END` if training
    duration >= `swa_start`.

    See the :doc:`Method Card </method_cards/swa>` for more details.

    Example:
        .. testcode::

            from composer.algorithms import SWA
            from composer.trainer import Trainer
            swa_algorithm = SWA(
                swa_start=0.8
            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[swa_algorithm],
                optimizers=[optimizer]
            )

    Args:
        swa_start (float): Fraction of training completed before stochastic weight
            averaging is applied. Defalt = ``0.8``.
        anneal_epochs (int, optional): Number of epochs over which to anneal SWA
            learning rate. Default = ``10``.
        swa_lr (float, optional): The final learning rate used for weight averaging
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
        should_start_swa = float(state.get_elapsed_duration()) >= self.swa_start
        return event == Event.EPOCH_END and should_start_swa

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if self.swa_scheduler is None:

            if self.swa_lr is None:
                if len(state.schedulers) != 1:
                    raise RuntimeError("SWA supports only one scheduler")
                scheduler = state.schedulers[0]
                last_lr = scheduler.get_last_lr()
                if len(last_lr) != 1:
                    raise RuntimeError(f"SWA supports only one LR; instead found {len(last_lr)}")
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
