# Copyright 2022  Gihyun Park, Junyeol Lee, and Jiwon Seo
# SPDX-License-Identifier: Apache-2.0
import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch
import numpy as np
from torch.optim import Optimizer

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


class GyroDropoutLayer(torch.nn.Module):
    def __init__(self, iters_per_epoch: int, max_epoch: int, p:float, sigma:int, tau:int):
        super(GyroDropoutLayer, self).__init__()    

        self.sigma = sigma
        self.tau = tau
        self.p = p
        self.preselect_masks = None
        self.dropout_mask = None
        self.training_step = 0
        self.iter_num = 0
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        
    def forward(self, x):
        if self.training:
            if self.training_step == 0:
                is_cuda_tensor = x.is_cuda

                if is_cuda_tensor:
                    self.preselect_masks = (torch.rand(self.sigma, x.shape[1]) > self.p).float().to("cuda")
                else:
                    self.preselect_masks = (torch.rand(self.sigma, x.shape[1]) > self.p).float()
                
                self.iter_num = int(self.iters_per_epoch * self.max_epoch / self.sigma) * self.tau
                # above simplified from: (iters_per_epoch*max_epoch*batch_size/sigma) / (batch_size/self.tau) 
            if self.training_step % self.iter_num == 0:
                pick_idx = np.random.choice(self.sigma, self.tau)
                self.selected_masks = self.preselect_masks[pick_idx]

            self.dropout_mask = torch.repeat_interleave(self.selected_masks, x.shape[0] // self.tau, dim=0)

            self.training_step += 1          

            return x * self.dropout_mask * (1 / (1 - self.p))
        else:
            return x


def from_Dropout(p: float, sigma: int, tau: int, num_iterations: int, epoch:int, layer: torch.nn.Module, module_index: int):
    """Defines a replacement policy from a `torch.nn.Dropout` to a 'GyroDropout`"""

    return GyroDropoutLayer(p, sigma, tau, num_iterations, epoch)


def apply_gyro_dropout(model: torch.nn.Module, 
                       iters_per_epoch: int,
                       max_epoch: int,
                       p: float,
                       sigma: int, 
                       tau: int, 
                       optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,) -> None:
    """Replaces all instances of `torch.nn.Dropout` with a `GyroDropout`.

    By masking Dropout layer, this usually improves accuracy.
    """  
    # prepare the replacement policy and perform replacement
    from functools import partial
    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.Dropout: partial(from_Dropout, iters_per_epoch, max_epoch, p, sigma, tau)}
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(
            NoEffectWarning(
                'No instances of `torch.nn.Dropout` were found, and therefore, there were no modules to replace.'))
    log.info(f'Successfully replaced {len(replaced_instances)} of dropout with a Gyro dropout.')


class GyroDropout(Algorithm):
    """Replaces all instances of `torch.nn.Dropout` with a `GyroDropout`.

    By masking Dropout layer, this usually improves accuracy.

    Args:
        iters_per_epoch (int): Total iterations per an epoch.
        max_epoch (int): Total epoches in training.
        p (float, optional): Float number of ratio to dropout.
            Default: ``0.5``.
        sigma (int, optional): the number of total pre-selected subnetwork
            Default: ``256``.
        tau (int, optional): the number of concurrently scheduled subnetworks in an iteration
            Default: ``16``.
            
    Example:
           
        .. testcode::

           from composer.algorithms import GyroDropout

           algorithm = GyroDropout(iters_per_epoch=196, max_epoch=100, p=0.5, sigma=256, tau=16)
           trainer = Trainer(
               model=model,
               train_dataloader=train_dataloader,
               max_duration="100ep",
               algorithms=[algorithm],
               optimizers=[optimizer]
           )
    """

    def __init__(self, iters_per_epoch: int, max_epoch: int, p: float=0.5, sigma: int=256, tau: int=16):  # XXX: add default faluse. change the order of arguments
        self.p = p 
        self.sigma = sigma
        self.tau = tau
        self.iters_per_epoch = iters_per_epoch
        self.max_epoch = max_epoch
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool: # XXX: not sure if this is correct.
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused   XXX: not sure if this is correct. Copied from FusedLayerNorm.
        apply_gyro_dropout(model=state.model,
                           optimizers=state.optimizers,
                           iters_per_epoch=self.iters_per_epoch,
                           max_epoch=self.max_epoch,
                           p=self.p,
                           sigma=self.sigma,
                           tau=self.tau,
        )
