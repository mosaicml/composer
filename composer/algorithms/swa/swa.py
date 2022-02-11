# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim.swa_utils import SWALR, AveragedModel

from composer.core.types import Algorithm, DataLoader, Event, Logger, State

log = logging.getLogger(__name__)


@torch.no_grad()
def update_bn(loader: DataLoader, model: torch.nn.Module, device: torch.device):
    """Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert module.running_mean is not None
            module.running_mean = torch.zeros_like(module.running_mean)
            assert module.running_var is not None
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for data in loader:
        import composer.trainer.devices as devices
        if device.type == "cuda":
            composer_device = devices.DeviceGPU()
        elif device.type == "cpu":
            composer_device = devices.DeviceCPU()
        else:
            raise ValueError("`device` must be one of 'cuda', 'cpu'")
        data = composer_device.batch_to_device(data)
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
        swa_start (float, optional): fraction of training completed before stochastic
            weight averaging is applied. Default = 0.85.
        swa_end (float, optional): fraction of training completed before stochastic weight averaging is
            completed. Default = 0.97
        swa_lr (float, optional): the final learning rate used for weight averaging

    Note that 'anneal_epochs' is not used in the current implementation
    """

    def __init__(self,
                 swa_start: float = 0.85,
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
        """Run on EPOCH_END if training duration is greater than `swa_start` and less than `swa_end`.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        should_start_swa = float(state.get_elapsed_duration()) >= self.swa_start and not self.swa_completed
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
