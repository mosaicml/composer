# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Source code is compiled from a modified version of:
# https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/modules/module.py
# Link to PyTorch License File: https://github.com/pytorch/pytorch/blob/master/LICENSE
# TODO: This code will need to be removed when PyTorch correctly supports delayed initialization
# with meta tensors.

"""Helper function to safely call .apply for initializing meta tensors in PyTorch."""

import torch
from torch.nn.parameter import Parameter


def meta_safe_apply(self, fn, ignored_modules: set, module_name: str):
    """Applies the function recursively to a module's children and the module itself.

    This variant allows us to ignore modules to apply the function.
    The function is a slightly modified version of the one from PyTorch:
    https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/modules/module.py#L637

    Args:
        self: the module to apply fn to.
        fn: the function called to each submodule
        ignored_modules: a set of names of modules to not apply fn.
        module_name: the current module's name.
    """
    for name, module in self.named_children():
        module_name_list = [module_name, name]
        if module_name == '':
            module_name_list = [name]
        curr_module_name = concatenate_strings(module_name_list)
        meta_safe_apply(module, fn, ignored_modules, curr_module_name)

    def compute_should_use_set_data(tensor, tensor_applied):
        if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
            # If the new tensor has compatible tensor type as the existing tensor,
            # the current behavior is to change the tensor in-place using `.data =`,
            # and the future behavior is to overwrite the existing tensor. However,
            # changing the current behavior is a BC-breaking change, and we want it
            # to happen in future releases. So for now we introduce the
            # `torch.__future__.get_overwrite_module_params_on_conversion()`
            # global flag to let the user control whether they want the future
            # behavior of overwriting the existing tensor or not.
            return not torch.__future__.get_overwrite_module_params_on_conversion()
        else:
            return False

    for key, param in self._parameters.items():
        curr_name = concatenate_strings([module_name, key])
        if param is None or curr_name in ignored_modules:
            continue
        # Tensors stored in modules are graph leaves, and we don't want to
        # track autograd history of `param_applied`, so we have to use
        # `with torch.no_grad():`
        with torch.no_grad():
            param_applied = fn(param)
        should_use_set_data = compute_should_use_set_data(param, param_applied)
        if should_use_set_data:
            param.data = param_applied
            out_param = param
        else:
            assert isinstance(param, Parameter)
            assert param.is_leaf
            out_param = Parameter(param_applied, param.requires_grad)
            self._parameters[key] = out_param

        if param.grad is not None:
            with torch.no_grad():
                grad_applied = fn(param.grad)
            should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
            if should_use_set_data:
                assert out_param.grad is not None
                out_param.grad.data = grad_applied
            else:
                assert param.grad.is_leaf
                out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

    for key, buf in self._buffers.items():
        if buf is not None:
            self._buffers[key] = fn(buf)

    return self


def concatenate_strings(str_list, delim='.'):
    """Concatenates a list of strings together with a delimiter in between the strings in the list.

    Args:
        str_list: a list of string to join.
        delim: the delimiter to separate all strings
    """
    return delim.join(str_list)
