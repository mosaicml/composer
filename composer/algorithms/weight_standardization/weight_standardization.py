# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

__all__ = ['apply_weight_standardization', 'WeightStandardization']


def _standardize_weights(W: torch.Tensor):
    reduce_dims = list(range(1, W.dim()))
    W_var, W_mean = torch.var_mean(W, dim=reduce_dims, keepdim=True, unbiased=False)
    return (W - W_mean) / (torch.sqrt(W_var + 1e-10))


class WeightStandardizer(nn.Module):

    def forward(self, W):
        return _standardize_weights(W)


def apply_weight_standardization(model: torch.nn.Module):
    count = 0
    for module in model.modules():
        if (isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):

            parametrize.register_parametrization(module, 'weight', WeightStandardizer())
            count += 1
    return count


class WeightStandardization(Algorithm):

    def __init__(self):
        pass

    def match(self, event: Event, state: State):
        return (event == Event.INIT)

    def apply(self, event: Event, state: State, logger: Logger):
        count = apply_weight_standardization(state.model)
        logger.data_fit({'WeightStandardization/num_weights_standardized': count})
