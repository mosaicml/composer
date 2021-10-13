# Copyright 2021 MosaicML. All Rights Reserved.

# type: ignore
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import SWALR, AveragedModel

from composer.algorithms.swa.hparams import SWAHparams
from composer.core.types import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)

import math

import torch
from torch.nn import Module


@torch.no_grad()
def update_bn(loader, model, device=None):
    """
    Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
    in order to work with our internal trainer.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for i, data in enumerate(loader):
        model(data)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


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
        self.hparams = SWAHparams(
            swa_start=swa_start,
            anneal_epochs=anneal_epochs,
            swa_lr=swa_lr,
        )
        assert 0 < swa_start < 1, "swa_start must be between 0 and 1."
        assert anneal_epochs > 0, "anneal_epochs must be great than 0."

        self.swa_scheduler = None

    def match(self, event: Event, state: State) -> bool:
        """Run in Event.TRAINING_START, Event.TRAINING_END or if Event.EPOCH_END and epochs greater than or equal to `swa_start * max_epochs`

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        should_start_swa = state.epoch >= int(self.hparams.swa_start * state.max_epochs)
        return event in (Event.TRAINING_START, Event.TRAINING_END) or \
             (event == Event.EPOCH_END and should_start_swa)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Apply SWA to weights towards the end of training

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.model is not None, 'We cannot apply SWA to None'

        swa_start_epochs = int(self.hparams.swa_start * state.max_epochs)

        if event == Event.TRAINING_START:
            self.swa_model = AveragedModel(state.model)

        if event == Event.EPOCH_END and state.epoch == swa_start_epochs:
            assert self.swa_scheduler is None, "SWA Scheduler should only be set once. Another algorithm "
            "may have adjusted the max_epochs."

            if self.hparams.swa_lr is None:
                last_lr = state.schedulers.schedulers[0].get_last_lr()  # assumes ComposedScheduler
                log.info(f'Setting SWA LR to {last_lr}')
                self.hparams.swa_lr = last_lr

            self.swa_scheduler = SWALR(
                state.optimizers[0] if isinstance(state.optimizers, tuple) else state.optimizers,
                swa_lr=self.hparams.swa_lr,
                anneal_epochs=self.hparams.anneal_epochs,
                anneal_strategy='cos',
            )

        if event == Event.EPOCH_END and state.epoch >= swa_start_epochs:
            self.swa_model.update_parameters(state.model)

            if self.swa_scheduler is None:
                raise ValueError('SWA LR scheduler was not set.')
            self.swa_scheduler.step()

        ## end of training
        if event == Event.TRAINING_END:
            update_bn(state.train_dataloader, self.swa_model)
            state.model = self.swa_model
            log.info('Updated BN and set model to the averaged model')
