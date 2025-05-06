# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import torch
from torch.distributed.tensor import DTensor

def meta_init(model: torch.nn.Module) -> None:
    """Initialize a module's parameters if it has `meta` Parameters or Buffers.

    This function initializes parameters for models with meta tensors by first converting
    them to empty tensors on CUDA, then applying initialization functions.
    
    Args:
        model (torch.nn.Module): The model containing meta parameters to initialize.
        
    Raises:
        ValueError: If the model doesn't have a `param_init_fn` or modules don't have
            `reset_parameters` methods for initialization.
            
    Notes:
        - This function only works with parameters that are DTensors.
        - It assumes buffers are not shared among modules.
        - It requires either a model-level `param_init_fn` or module-level
          `reset_parameters` methods to perform the actual initialization.
    """

    param_init_fn = getattr(model, 'param_init_fn', None)
    for module in model.modules():
        has_meta = any(param.is_meta for param in module.parameters(recurse=False)) or any(buffer.is_meta for buffer in module.buffers(recurse=False))
        if not has_meta:
            continue
        for param in module.parameters(recurse=False):
            # NOTE we assume buffers are not shared among modules
            assert isinstance(param, DTensor), 'Only DTensor can be meta initialized safely w/o breaking potential weight tying'
        module.to_empty(device='cuda', recurse=False)
        if isinstance(param_init_fn, Callable):
            param_init_fn(module)
        elif isinstance(getattr(module, 'reset_parameters', None), Callable):
            module.reset_parameters()
        else:
            raise ValueError(
                f'Model `{model}` does not have a ``param_init_fn`` or submodule `{module}` does not have a ``reset_parameters`` function. '
                'This leaves parameters without initialization. Please add a ``param_init_fn`` or ``reset_parameters`` '
                f'to root model `{model}` or submodule `{module}`.',
            )
