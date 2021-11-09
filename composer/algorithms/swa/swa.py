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
        """Run in Event.TRAINING_START or if Event.EPOCH_END and steps greater than or equal to `swa_start * max_epochs * steps_per_epoch`

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """

        if state.max_epochs == 1 and event == Event.TRAINING_START:
            should_start_swa = state.step >= int(self.hparams.swa_start * state.steps_per_epoch)
            return (event == Event.TRAINING_START or should_start_swa)

        else:
            if event == Event.TRAINING_START:
                should_start_swa = state.step >= int(self.hparams.swa_start * state.max_epochs * state.steps_per_epoch)

            return (event == Event.TRAINING_START) or \
                (event == Event.EPOCH_END and should_start_swa)

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        """Apply SWA to weights towards the end of training

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.model is not None, 'We cannot apply SWA to None'

        swa_start_steps = state.step >= int(self.hparams.swa_start * state.max_epochs * state.steps_per_epoch)

        if event == Event.TRAINING_START:
            self.swa_model = AveragedModel(state.model.module)
            log.info("Creating AveragedModel")

        if self.hparams.swa_lr is None:
            log.info(f'Setting SWA LR to default lr')

        if swa_start_steps:
            self.swa_model.update_parameters(state.model)
            log.info("updating AveragedModel")

            if state.max_epochs == 1:
                update_bn(state.train_dataloader, self.swa_model)
                state.model.module = self.swa_model.module
                log.info('Updated BN and set model to the averaged model')

            else:
                if event == Event.EPOCH_END and state.epoch == state.max_epochs - 1:
                    update_bn(state.train_dataloader, self.swa_model)
                    state.model.module = self.swa_model.module
                    log.info('Updated BN and set model to the averaged model')
