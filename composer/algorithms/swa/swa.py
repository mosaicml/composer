# Copyright 2021 MosaicML. All Rights Reserved.

"""Core code for Stochastic Weight Averaging."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim.swa_utils import SWALR, AveragedModel

from composer.core.types import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)

__all__ = ['SWA']


class SWA(Algorithm):
    """Apply Stochastic Weight Averaging (`Izmailov et al., 2018 <https://arxiv.org/abs/1803.05407>`_)

    Stochastic Weight Averaging (SWA) averages model weights sampled at
    different times near the end of training. This leads to better
    generalization than just using the final trained weights.

    Because this algorithm needs to maintain both the current value of the
    weights and the average of all of the sampled weights, it doubles the
    model's memory consumption. Note that this does not mean that the total
    memory required doubles, however, since stored activations and the
    optimizer state are not doubled.

    Uses PyTorch's :mod:`torch.optim.swa_utils` under the hood.

    See the :doc:`Method Card </method_cards/swa>` for more details.

    Example:
        .. testcode::

            from composer.algorithms import SWA
            from composer.trainer import Trainer
            swa_algorithm = SWA(
                swa_start=0.7,
                swa_end=0.8
            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="10ep",
                algorithms=[swa_algorithm],
                optimizers=[optimizer]
            )

    Args:
        swa_start (float, optional): Fraction of training completed before stochastic
            weight averaging begins. Defalt = ``0.7``.
        swa_end (float, optional): Fraction of training completed before the baseline
            (non-averaged) model is replaced with the stochastic weight averaged model.
            It's important to have at least one epoch of training after the baseline model
            is replaced by the SWA model so that the SWA model can have its buffers (most
            importantly its batch norm statistics) updated. If ``swa_end`` occurs during
            the final epoch of training (e.g. ``swa_end = 0.9`` and ``max_duration =
            "5ep"``), the SWA model will not have its buffers updated, which can
            negatively impact accuracy, so ensure ``swa_end`` <
            :math:`frac{N_{epochs}-1}{N_{epochs}}. Default = ``0.97``.
        schedule_swa_lr (bool, optional): Flag to determine whether apply an SWA-specific
            LR schedule during the period in which SWA is active. Default = ``False``.
        anneal_strategy (str, optional): SWA learning rate annealing schedule strategy.
            "linear" for linear annealing, "cos" for cosine annealing. Default =
            ``"linear"``.
        anneal_epochs (int, optional): Number of epochs over which to anneal SWA
            learning rate. Default = ``10``.
        swa_lr (float, optional): The final learning rate for the SWA LR schedule.
    """

    def __init__(self,
                 swa_start: float = 0.7,
                 swa_end: float = 0.97,
                 anneal_epochs: int = 10,
                 swa_lr: Optional[float] = None):
        self.swa_start = swa_start
        self.swa_end = swa_end
        self.anneal_epochs = anneal_epochs
        self.swa_lr = swa_lr
        self.swa_model: Optional[torch.nn.Module] = None
        self.swa_completed = False

        assert 0 < swa_start < 1, "swa_start must be between 0 and 1."
        assert swa_end <= 1, "swa_end must be ≤ 1"
        assert anneal_epochs > 0, "anneal_epochs must be great than 0."

        self.swa_scheduler = None
        self.swa_model = None

    def match(self, event: Event, state: State) -> bool:
        should_start_swa = float(state.get_elapsed_duration()) >= self.swa_start and not self.swa_completed
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
        # self.swa_scheduler.step()

        ## end of swa window
        if float(state.get_elapsed_duration()) >= self.swa_end:
            # device = next(self.swa_model.parameters()).device
            # TODO(laura) this does not apply the batch split fn. This may result in cuda OOM.
            # update_bn(state.train_dataloader, model=self.swa_model, device=device)
            assert type(self.swa_model.module) == type(state.model)
            state.model.load_state_dict(self.swa_model.module.state_dict())  # type: ignore
            # log.info('Updated BN and set model to the averaged model')
            self.swa_completed = True
            log.info('Set model to the averaged model')
