# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
"""FX-based model transformation and optimization. 

Provides utilities to do FX-based model transfomrations. 
"""

import logging
import torch
import torch.nn as nn
from typing import Union
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule


log = logging.getLogger(__name__)

__all__ = ["trace_with_fx"]


def trace_with_fx(
    module: nn.Module,
) -> Union[GraphModule, nn.Module]:
    """Trace a module with the default FX tracer.

    Arguments:
        module (torch.nn.Module): Model to trace

    """

    try:
        return symbolic_trace(module)
    except Exception as e:
        log.warning(
            f"Tracing with torch.fx failed with the following exception! Continuing training with the original nn.Module."
            f" {str(e)}"
        )
        return module

def count_operator_instances(gm: GraphModule, operator) -> int:
    """Counts the number of instances of ``operator`` in ``gm``.

    .. rubric:: Example

    .. testsetup::

        from composer.utils.fx_utils import count_operator_instances

    .. doctest::

        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 64), nn.ReLU())
        >>> count_module_instances(module, nn.Linear)
        2
        >>> count_module_instances(module, (nn.Linear, nn.ReLU))
        3

    Args:
        module (torch.nn.Module): The source module.
        module_class (Type[torch.nn.Module] | Tuple[Type[torch.nn.Module], ...]):
            The module type (or tuple of module types) to count.

    Returns:
        int: The number of instances of ``operator`` in ``gm``
    """
    count = 0
    for _, child in module.named_children():
        if isinstance(child, module_class):
            count += 1
        count += count_module_instances(child, module_class)

    return count

def replace_operator(
    module: GraphModule,
):
    """Replace a single torch method or function with another.
    Arguments:

    """
    pass


def replace_pattern(module: GraphModule):
    """Search and replace the pattern with another.
    Arguments:
    """
    pass


def detect_residual_pattern(module: GraphModule):
    """Search and replace the pattern with another.
    Arguments:
    """
    pass


def replace_residual_with_stochastic(module: GraphModule):
    """Replaces residual pattern with their stoachstic equivalent.
    Arguments:
    """
    pass


def fuse_parallel_linears(module: GraphModule):
    """If there are parallel linears in the model, fuse them together.
    Arguments:
    """
    pass
