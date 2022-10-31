# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch
import numpy as np

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)

class GyroDropoutLayer(torch.nn.Module):
    #def __init__(self, in_features, num_subnets=16, num_masks=256, masks_per_batch=16, p=0.5):
    def __init__(self, sigma=256, tau=16, p=0.5, in_features=1024):
        super(GyroDropoutLayer, self).__init__()

        self.sigma = sigma
        self.tau = tau
        self.p = p
        self.mask_list = None
        self.dropout_mask = None
        self.training_step = 0
        self.training_period = 0
        self.training_epoch = 100
        self.num_iterations = 12500
        
    def forward(self, x):

        if self.training:
            if self.training_step == 0:
                # init
                is_cuda_tensor = x.is_cuda
                if is_cuda_tensor:
                    # mask_list is gpu tensor
                    self.mask_list = (torch.rand(self.sigma, x.shape[1]) > self.p).float().to("cuda")
                else:
                    # mask_list is cpu tensor
                    self.mask_list = (torch.rand(self.sigma, x.shape[1]) > self.p).float()
                
                self.training_period = int(self.num_iterations * self.training_epoch / self.sigma) * self.tau
            ##여기부터 원래 forward 코드
            if self.training_step % self.training_period == 0:
                pick_idx = np.random.choice(self.sigma, self.tau)
                picked_subnets = self.mask_list[pick_idx]
                self.dropout_mask = torch.repeat_interleave(picked_subnets, x.shape[0] // self.tau, dim=0)


            self.training_step += 1          
            # print(x.size())
            # print(self.dropout_mask.size())
            # print(1/(1-self.p))
            return x
            #return x * self.dropout_mask * (1 / (1 - self.p))
        else:
            return x

def from_LayerNorm(sigma: int, tau: int, layer: torch.nn.Module, module_index: int):
    """Defines a replacement policy from a `torch.nn.LayerNorm` to a `apex.normalization.fused_layer_norm`"""
    # print()
    # print("GGG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #raise RuntimeError('B is not a rank 2 tensor: has shape')
    return GyroDropoutLayer(sigma, tau)

def apply_gyro_dropout(model: torch.nn.Module, optimizers: Union[torch.optim.Optimizer,
                                                                    Sequence[torch.optim.Optimizer]]
                                                                    ,sigma, tau) -> None:
    """Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedLayerNorm
    <https://nvidia.github.io/apex/layernorm.html>`_.

    By fusing multiple kernel launches into one, this usually improves GPU utilization.
    """
    
    ##print(State.max_duration)
    #sys.exit()
    
    # prepare the replacement policy and perform replacement
    from functools import partial
    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {torch.nn.Dropout: partial(from_LayerNorm, sigma, tau)}
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(
            NoEffectWarning(
                'No instances of `torch.nn.Dropout` were found, and therefore, there were no modules to replace.'))
    log.info(f'Successfully replaced {len(replaced_instances)} of dropout with a Gyro dropout.')


class GyroDropout(Algorithm):
    """Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedLayerNorm
    <https://nvidia.github.io/apex/layernorm.html>`_.

    By fusing multiple kernel launches into one, this usually improves GPU utilization.

    Runs on ``Event.INIT``, so it can replace all instances of `torch.nn.LayerNorm` before the model is DDP wrapped. Has no hyperparameters.

    Example:
        .. testsetup::

           def no_op(self, *args): pass

           from composer.algorithms import FusedLayerNorm

           FusedLayerNorm.__init__ = no_op

           FusedLayerNorm.apply = no_op

           model, train_dataloader, optimizer = _make_synthetic_bert_state()

        .. testcode::

           from composer.algorithms import FusedLayerNorm

           algorithm = FusedLayerNorm()
           trainer = Trainer(
               model=model,
               train_dataloader=train_dataloader,
               max_duration="1ep",
               algorithms=[algorithm],
               optimizers=[optimizer]
           )
    """

    def __init__(self, sigma, tau):
        # GyroDropout takes 2 arguments - sigma, tau
        
        self.sigma = sigma
        self.tau = tau

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_gyro_dropout(model=state.model, optimizers=state.optimizers, sigma=self.sigma, tau=self.tau)
