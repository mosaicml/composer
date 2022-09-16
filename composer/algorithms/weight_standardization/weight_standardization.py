# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.fx import symbolic_trace

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

__all__ = ['apply_weight_standardization', 'WeightStandardization']


def _standardize_weights(W: torch.Tensor):
    reduce_dims = list(range(1, W.dim()))
    W_var, W_mean = torch.var_mean(W, dim=reduce_dims, keepdim=True, unbiased=False)
    return (W - W_mean) / (torch.sqrt(W_var + 1e-5))


class WeightStandardizer(nn.Module):

    def forward(self, W):
        return _standardize_weights(W)


def apply_weight_standardization(model: torch.nn.Module, n_last_layers_ignore: bool = False):
    count = 0
    model_trace = symbolic_trace(model)
    for module in model_trace.modules():
        if (isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
            parametrize.register_parametrization(module, 'weight', WeightStandardizer())
            count += 1

    target_ws_layers = count - n_last_layers_ignore

    for module in list(model_trace.modules())[::-1]:
        if target_ws_layers == count:
            break
        if (isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized=False)
            count -= 1

    return count


class WeightStandardization(Algorithm):
    """Weight standardization.

    Weight standardization.

    Args:
        ignore_last_layer (bool): ignores the laste layer. Default: ``False``.
    """

    # TODO: Maybe make this ignore last n layers in case there are multiple prediction heads? Would this work?
    def __init__(self, n_last_layers_ignore: bool = False):
        self.n_last_layers_ignore = n_last_layers_ignore

    def match(self, event: Event, state: State):
        return (event == Event.INIT)

    def apply(self, event: Event, state: State, logger: Logger):
        count = apply_weight_standardization(state.model, n_last_layers_ignore=self.n_last_layers_ignore)
        logger.data_fit({'WeightStandardization/num_weights_standardized': count})
