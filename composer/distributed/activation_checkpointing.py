# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for activation checkpointing. Note that while this is orthogonal to FSDP2, it is implemented in the distributed directory because it is closely related to FSDP2."""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
    CheckpointImpl,
    OffloadWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)

from composer.utils.parallelism import FSDP2Config


def validate_activation_wrapper(
    module: torch.nn.Module,
    is_activation_checkpoint_enabled: bool,
    is_activation_offload_enabled: bool,
    checkpoint_fn: Optional[Callable] = None,
) -> None:
    """Verify that activation checkpointing and offload wrappers exist where expected based on the boolean flags.

    Raises ValueError if validation fails, listing the offending module names.

    Args:
        module (torch.nn.Module): The root model module to inspect.
        is_activation_checkpoint_enabled (bool): Whether activation checkpointing wrappers (`ActivationWrapper`) are expected.
        is_activation_offload_enabled (bool): Whether activation offload wrappers (`OffloadWrapper`) are expected.
        checkpoint_fn (Optional[Callable]): An optional function to determine if a module should be checkpointed.
    """
    offenders: list[str] = []

    def check_module_needs_wrapping(inner_module: torch.nn.Module) -> bool:
        if checkpoint_fn is not None and checkpoint_fn(inner_module):
            return True
        return hasattr(inner_module, '_activation_checkpointing') and bool(inner_module._activation_checkpointing)

    def _walk(mod: torch.nn.Module, prefix: str) -> None:
        inner_mod = mod
        actual_is_offloaded = isinstance(mod, OffloadWrapper)
        actual_is_checkpointed = False

        if actual_is_offloaded:
            offload_inner = getattr(mod, _CHECKPOINT_WRAPPED_MODULE, None)
            if offload_inner is None:
                # We shouldn't ever see this, but it's good to have the check
                offenders.append(f'{prefix}: OffloadWrapper missing inner module')
                return

            actual_is_checkpointed = isinstance(offload_inner, ActivationWrapper)
            if actual_is_checkpointed:
                inner_mod = getattr(offload_inner, _CHECKPOINT_WRAPPED_MODULE, None)
                if inner_mod is None:
                    # We shouldn't ever see this, but it's good to have the check
                    offenders.append(f'{prefix}: ActivationWrapper missing inner module')
                    return
            else:
                inner_mod = offload_inner
        elif isinstance(mod, ActivationWrapper):
            actual_is_checkpointed = True
            inner_mod = getattr(mod, _CHECKPOINT_WRAPPED_MODULE, None)
            if inner_mod is None:
                # We shouldn't ever see this, but it's good to have the check
                offenders.append(f'{prefix}: ActivationWrapper missing inner module')
                return

        module_needs_wrapping = check_module_needs_wrapping(inner_mod)

        validation_failed = False
        error_reasons = []
        if module_needs_wrapping:
            if actual_is_checkpointed != is_activation_checkpoint_enabled:
                validation_failed = True
                error_reasons.append(
                    f'Expected checkpointed={is_activation_checkpoint_enabled}, Got={actual_is_checkpointed}',
                )
            if actual_is_offloaded != is_activation_offload_enabled:
                validation_failed = True
                error_reasons.append(f'Expected offloaded={is_activation_offload_enabled}, Got={actual_is_offloaded}')
        else:
            if actual_is_checkpointed:
                validation_failed = True
                error_reasons.append('Should not be checkpointed, but is')
            if actual_is_offloaded:
                validation_failed = True
                error_reasons.append('Should not be offloaded, but is')

        if validation_failed:
            offenders.append(f"{prefix or inner_mod.__class__.__name__}: {', '.join(error_reasons)}")

        # Recurse on the children of the *original* module structure
        for name, child in inner_mod.named_children():
            child_prefix = f'{prefix}.{name}' if prefix else name
            _walk(child, child_prefix)

    _walk(module, '')
    if len(offenders) > 0:
        raise ValueError(f'Activation wrapper validation failed. Offending modules: {offenders}')


def generate_default_check_fn(model: nn.Module) -> Callable:
    """Generates the default check fn for activation checkpointing/offloading."""

    def _check_fn(module: torch.nn.Module) -> bool:
        if hasattr(module, '_activation_checkpointing'):
            return bool(module._activation_checkpointing)
        if hasattr(
            model,
            'activation_checkpointing_fn',
        ) and isinstance(model.activation_checkpointing_fn, Callable):
            return model.activation_checkpointing_fn(module)
        return False

    return _check_fn


def apply_ac(model: nn.Module, fsdp2_config: FSDP2Config, check_fn: Optional[Callable] = None) -> None:
    """Apply activation checkpointing to the model. This is orthogonal to FSDP2 so it can be applied pre-sharding or post-sharding.

    This method follows the same logic as FSDP1 as well as TorchTitan's AC example. FSDP1 doesn't support op-based
    selective activation checkpointing, so that will not be implemented unless we get sufficient demand.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        fsdp2_config (FSDP2Config): The FSDP2 configuration.
        check_fn (Optional[Callable]): An optional function to determine if a module should be checkpointed.
    """
    activation_checkpointing = fsdp2_config.activation_checkpointing
    activation_cpu_offload = fsdp2_config.activation_cpu_offload

    # Create the base checkpointing wrapper using no_reentrant checkpointing by default as
    # PyTorch notes that reentrant checkpointing is deprecated and will be removed in a future release
    opt_checkpoint_wrapper = lambda m: checkpoint_wrapper(
        m,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ) if activation_checkpointing else (lambda module: module)
    # Create the combined wrapper which takes cpu offloading into consideration
    opt_combined_wrapper = (
        lambda module: offload_wrapper(
            opt_checkpoint_wrapper(module)
            if activation_checkpointing else module,  # type: ignore reportGeneralTypeIssues
        )
    ) if activation_cpu_offload else opt_checkpoint_wrapper

    # Create the check function to determine if a module should be checkpointed
    if check_fn is None:
        check_fn = generate_default_check_fn(model)

    # Apply the activation checkpointing on the model, this uses _recursive_wrap to apply the wrapper to all submodules
    # but doesn't apply the wrapper to the root module
    apply_activation_checkpointing(model, opt_combined_wrapper, check_fn)  # type: ignore

    # Validate that activation checkpointing is applied correctly
    validate_activation_wrapper(model, activation_checkpointing, activation_cpu_offload, check_fn)
