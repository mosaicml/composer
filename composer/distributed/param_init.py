# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Param initialization helper functions."""

from typing import Callable, Optional

import torch.nn as nn
from torch.distributed.tensor import DTensor


def meta_init(module: nn.Module, param_init_fn: Optional[Callable[[nn.Module], None]] = None) -> None:
    """Initialize a module's parameters if it has `meta` Parameters or Buffers.

    This function initializes parameters for models with meta tensors by first converting
    them to empty tensors on CUDA, then applying initialization functions.

    Args:
        module (torch.nn.Module): The module containing meta parameters to initialize.
        param_init_fn (Optional[Callable[[nn.Module], None]]): An optional function to initialize
            the module's parameters. If not provided, the function will look for a `param_init_fn`
            attribute on the module or use the module's `reset_parameters` method.

    Raises:
        ValueError: If the module doesn't have a `param_init_fn` or modules don't have
            `reset_parameters` methods for initialization.

    Notes:
        - This function only works with parameters that are DTensors.
        - It assumes buffers are not shared among modules.
        - It requires either a module-level `param_init_fn` or module-level
          `reset_parameters` methods to perform the actual initialization.
    """
    # TODO need a unit test for this
    cur_param_init_fn = getattr(module, 'param_init_fn', param_init_fn)
    for child in module.children():
        meta_init(child, cur_param_init_fn)
    has_meta = any(param.is_meta for param in module.parameters(recurse=False)
                  ) or any(buffer.is_meta for buffer in module.buffers(recurse=False))
    if not has_meta:
        return
    for param in module.parameters(recurse=False):
        # NOTE we assume buffers are not shared among modules
        assert isinstance(
            param,
            DTensor,
        ), 'Only DTensor can be meta initialized safely w/o breaking potential weight tying'
    module.to_empty(device='cuda', recurse=False)
    if isinstance(cur_param_init_fn, Callable):
        cur_param_init_fn(module)
    elif hasattr(module, 'reset_parameters') and isinstance(module.reset_parameters, Callable):
        module.reset_parameters()
    else:
        raise ValueError(
            f'Module `{module}` does not have (or have inherited) a ``param_init_fn`` or does not have a ``reset_parameters`` function. '
            'This leaves parameters without initialization. Please add a ``param_init_fn`` or ``reset_parameters`` '
            f'to module `{module}`.',
        )
