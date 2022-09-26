# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
import textwrap

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.fx import symbolic_trace

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)

__all__ = ['apply_weight_standardization', 'WeightStandardization']


def _standardize_weights(W: torch.Tensor):
    """Function to standardize the input weight ``W``"""
    reduce_dims = list(range(1, W.dim()))
    W_var, W_mean = torch.var_mean(W, dim=reduce_dims, keepdim=True, unbiased=False)
    return (W - W_mean) / (torch.sqrt(W_var + 1e-10))


class WeightStandardizer(nn.Module):
    """Class used to apply weight standardization with torch's parametrization."""

    def forward(self, W):
        return _standardize_weights(W)


def apply_weight_standardization(module: torch.nn.Module, n_last_layers_ignore: int = 0):
    """`Weight Standardization <https://arxiv.org/abs/1903.10520>`_ standardizes convolutional weights in a model.

    Args:
        module (torch.nn.Module): the torch module whose convolutional layers will be reparametrized.
        n_last_layers_ignore (int, optional): the number of layers at the end of the module to not apply weight standardization.
            Default: ``0``.
    """
    modules_to_reparametrize = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    # Attempt to symbolically trace a module, so the results of .modules() will be in the order of execution
    try:
        module_trace = symbolic_trace(module)
    except:
        if n_last_layers_ignore > 0:
            log.warning(
                textwrap.dedent(f"""\
                Model could not be symbolically traced. Modules ignored due to n_last_layers={n_last_layers_ignore} may
                not be the actual last layers of the network"""))
        module_trace = module

    # Count the number of convolution modules in the model
    conv_count = module_surgery.count_module_instances(module_trace, modules_to_reparametrize)

    # Calculate how many convs to reparametrize based on conv_count and n_last_layers_ignore
    target_ws_count = max(conv_count - n_last_layers_ignore, 0)

    # Reparametrize conv modules to use weight standardization
    current_ws_count = 0
    for m in module_trace.modules():
        # If the target number of weight standardized layers is reached, end for loop
        if current_ws_count == target_ws_count:
            break

        if isinstance(m, modules_to_reparametrize):
            parametrize.register_parametrization(m, 'weight', WeightStandardizer())
            current_ws_count += 1

    return current_ws_count


class WeightStandardization(Algorithm):
    """`Weight Standardization <https://arxiv.org/abs/1903.10520>`_ standardizes convolutional weights in a model.

    Args:
        n_last_layers_ignore (int, optional): the number of layers at the end of the model to not apply weight standardization.
            Default: ``0``.
    """

    def __init__(self, n_last_layers_ignore: int = 0):
        self.n_last_layers_ignore = n_last_layers_ignore

    def match(self, event: Event, state: State):
        return (event == Event.INIT)

    def apply(self, event: Event, state: State, logger: Logger):
        count = apply_weight_standardization(state.model, n_last_layers_ignore=self.n_last_layers_ignore)
        logger.data_fit({'WeightStandardization/num_weights_standardized': count})
