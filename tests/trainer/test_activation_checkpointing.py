# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Test activation checkpointing and offloading. Note that currently, this is only testing support for FSDP2 + activation checkpointing/offloading."""

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
    OffloadWrapper,
)
from tests.common import (
    ComposerCounterModel,
    CountModule,
    world_size,
)
from tests.trainer.test_fsdp2 import create_trainer_with_model
from tests.trainer.fsdp2_context import (
    fsdp2_context,
)

@world_size(2)
@pytest.mark.gpu
@fsdp2_context
@pytest.mark.parametrize(
    'activation_checkpointing,expected_forward_count,activation_cpu_offload',
    [
        (True, 2, True),
        (True, 2, False),
        (False, 1, True),
        (False, 1, False),
    ],
)
def test_fsdp2_activation_checkpointing_attribute(
    world_size: int,
    activation_checkpointing: bool,
    expected_forward_count: int,
    activation_cpu_offload: bool,
):
    """Test FSDP2 activation checkpointing."""
    del world_size

    model = ComposerCounterModel(num_inputs=10, num_outputs=10, device='cuda')
    if activation_checkpointing or activation_cpu_offload:
        model.module[0]._activation_checkpointing = True  # type: ignore

    # Train the model on one batch to make sure forward is called the expected number of times
    trainer = create_trainer_with_model(
        model=model,
        num_classes=10,
        use_fsdp2=True,
        activation_checkpointing=activation_checkpointing,
        activation_cpu_offload=activation_cpu_offload,
        max_duration='1ba',
    )

    # Validate that the activation checkpointing wrapper is applied correctly pre-training
    # Note that the apply_ac function already validates the wrapper setup; we're just formalizing this
    checkpointed_module = trainer.state.model.module[0]
    if activation_cpu_offload:
        assert isinstance(checkpointed_module, OffloadWrapper), 'Expected OffloadWrapper for offloaded module'
        checkpointed_module = getattr(checkpointed_module, _CHECKPOINT_WRAPPED_MODULE)
    if activation_checkpointing:
        assert isinstance(checkpointed_module, ActivationWrapper), 'Expected ActivationWrapper for checkpointed module'
    non_checkpointed_module = trainer.state.model.module[-1]
    assert not isinstance(non_checkpointed_module, ActivationWrapper), 'Expected non-checkpointed module to not be an ActivationWrapper'
    assert not isinstance(non_checkpointed_module, OffloadWrapper), 'Expected non-offloaded module to not be an OffloadWrapper'

    trainer.fit()

    # validate that the activation checkpointing wrapper is applied correctly post-training (to make sure no side effects occur)
    # Note that the apply_ac function already validates the wrapper setup; we're just formalizing this
    module_to_test = trainer.state.model.module[0]
    if activation_cpu_offload:
        assert isinstance(module_to_test, OffloadWrapper), 'Expected OffloadWrapper'
        module_to_test = getattr(module_to_test, _CHECKPOINT_WRAPPED_MODULE)
    if activation_checkpointing:
        assert isinstance(module_to_test, ActivationWrapper), 'Expected ActivationWrapper'
    assert not isinstance(non_checkpointed_module, ActivationWrapper), 'Expected non-checkpointed module to not be an ActivationWrapper'
    assert not isinstance(non_checkpointed_module, OffloadWrapper), 'Expected non-offloaded module to not be an OffloadWrapper'

    error_msg = 'forward hook called {actual_forward_count} times, but expected {expected_forward_count} times.'
    counter_module_0_call_count = model.module[0].call_count  # type: ignore
    counter_module_1_call_count = model.module[-1].call_count  # type: ignore
    assert counter_module_0_call_count == expected_forward_count, \
        error_msg.format(expected_forward_count=expected_forward_count, actual_forward_count=counter_module_0_call_count)
    assert counter_module_1_call_count == 1, 'Expected last module to be called once since it is not checkpointed'


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
@pytest.mark.parametrize(
    'activation_checkpointing,expected_forward_count,activation_cpu_offload',
    [
        (True, 2, True),
        (True, 2, False),
        (False, 1, True),
        (False, 1, False),
    ],
)
def test_fsdp2_activation_checkpointing_fn(
    world_size: int,
    activation_checkpointing: bool,
    expected_forward_count: int,
    activation_cpu_offload: bool,
):
    """Test FSDP2 activation checkpointing."""
    del world_size

    model = ComposerCounterModel(num_inputs=10, num_outputs=10, device='cuda')
    activation_checkpointing_fn = None  # type: ignore

    # Checkpoint both CountModules
    if activation_checkpointing or activation_cpu_offload:

        def activation_checkpointing_fn(module: torch.nn.Module) -> bool:
            return isinstance(module, CountModule)

        model.module.activation_checkpointing_fn = activation_checkpointing_fn  # type: ignore

    # Train the model on one batch to make sure forward is called the expected number of times
    trainer = create_trainer_with_model(
        model=model,
        num_classes=10,
        use_fsdp2=True,
        activation_checkpointing=activation_checkpointing,
        activation_cpu_offload=activation_cpu_offload,
        max_duration='1ba',
    )
    # Validate that the activation checkpointing wrapper is applied correctly pre-training
    # Note that the apply_ac function already validates the wrapper setup; we're just formalizing this
    checkpointed_modules = [trainer.state.model.module[0], trainer.state.model.module[-1]]
    for module in checkpointed_modules:
        if activation_cpu_offload:
            assert isinstance(module, OffloadWrapper), 'Expected OffloadWrapper for offloaded module'
            module = getattr(module, _CHECKPOINT_WRAPPED_MODULE)
        if activation_checkpointing:
            assert isinstance(module, ActivationWrapper), 'Expected ActivationWrapper for checkpointed module'

    trainer.fit()

    # validate that the activation checkpointing wrapper is applied correctly post-training (to make sure no side effects occur)
    # Note that the apply_ac function already validates the wrapper setup; we're just formalizing this
    for module in checkpointed_modules:
        if activation_cpu_offload:
            assert isinstance(module, OffloadWrapper), 'Expected OffloadWrapper for offloaded module'
            module = getattr(module, _CHECKPOINT_WRAPPED_MODULE)
        if activation_checkpointing:
            assert isinstance(module, ActivationWrapper), 'Expected ActivationWrapper for checkpointed module'


    error_msg = 'forward hook called {actual_forward_count} times, but expected {expected_forward_count} times.'
    counter_module_0_call_count = model.module[0].call_count  # type: ignore
    counter_module_1_call_count = model.module[-1].call_count  # type: ignore
    assert counter_module_0_call_count == expected_forward_count, \
        error_msg.format(expected_forward_count=expected_forward_count, actual_forward_count=counter_module_0_call_count)
    assert counter_module_1_call_count == expected_forward_count, \
        error_msg.format(expected_forward_count=expected_forward_count, actual_forward_count=counter_module_1_call_count)


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
@pytest.mark.parametrize('type_of_checkpointing', ['attribute', 'function'])
def test_activation_checkpointing_cuda_memory_usage(world_size: int, type_of_checkpointing: str):
    """Test FSDP2 activation checkpointing CUDA memory usage."""
    del world_size

    cuda_memory = {}
    for activation_checkpointing in [True, False]:
        for activation_cpu_offload in [True, False]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Defining a large model to make the differences in memory usage clear
            model = ComposerCounterModel(num_inputs=1024, num_outputs=1024, num_hidden_layer_features=1024, device='cuda')

            # Testing out both ways to set the activation checkpointing
            if type_of_checkpointing == 'attribute':
                model.module[0].activation_checkpointing = activation_checkpointing
            elif type_of_checkpointing == 'function':
                def activation_checkpointing_fn(module: torch.nn.Module) -> bool:
                    return isinstance(module, CountModule)
                model.module.activation_checkpointing_fn = activation_checkpointing_fn
            else:
                raise ValueError(f"Invalid type of checkpointing: {type_of_checkpointing}")
            
            # Create trainer, run training, and check memory usage
            trainer = create_trainer_with_model(
                model=model,
                num_classes=1024,
                use_fsdp2=True,
                activation_checkpointing=activation_checkpointing,
                activation_cpu_offload=activation_cpu_offload,
                max_duration='1ba',
            )
            trainer.fit()
            peak_memory = torch.cuda.max_memory_allocated()
            cuda_memory[(activation_checkpointing, activation_cpu_offload)] = peak_memory
    
    # Assert that the memory usage is as expected

    base_mem = cuda_memory[(False, False)]
    ckpt_mem = cuda_memory[(True, False)]
    offload_mem = cuda_memory[(False, True)]

    # Checkpointing only should use less memory than base
    assert ckpt_mem < base_mem, "Checkpointing should use less memory than baseline"

    # Offloading only should use less memory than base
    assert offload_mem < base_mem, "Offloading should use less memory than baseline"

    # Checkpointing and offloading should use less memory than base
    both_mem = cuda_memory[(True, True)]
    assert both_mem < base_mem, "Checkpointing and Offloading should use less memory than baseline"

    # Checkpointing and offloading should ideally use less memory than checkpointing alone
    assert both_mem <= ckpt_mem, "Checkpointing and Offloading should use less memory than Checkpointing alone"

    # Checkpointing and offloading should ideally use less memory than offloading alone
    assert both_mem <= offload_mem, "Checkpointing and Offloading should use less memory than Offloading alone"
