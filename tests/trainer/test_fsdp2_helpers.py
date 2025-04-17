# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import pytest
import torch
import torch.nn as nn

from tests.trainer.fsdp2_context import (
    fsdp2_context,
    get_standalone_and_tied_modules,
    legalize_param_sharing_between_modules,
    get_valid_modules_to_shard,
)


def _context(func: Callable) -> Optional[Callable]:
    """Decorator to run tests with models initialized on the meta device for torch version 2.6+."""

    @fsdp2_context
    def wrapper(*args, **kwargs):
        with torch.device('meta'):
            return func(*args, **kwargs)

    return wrapper


class ModuleWithTiedParams(nn.Module):

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        # Tie weights
        self.linear2.weight = self.linear1.weight

    def forward(self, x):
        return self.linear2(self.linear1(x))


class MultiLinearTiedModel(nn.Module):
    """Base class for root models with three linear modules."""

    def __init__(self, in_features=10, out_features=20, share_weights=False):
        super().__init__()
        self.module1 = nn.Linear(in_features, out_features)
        self.module2 = nn.Linear(out_features, 10)
        self.module3 = nn.Linear(in_features, out_features)

        # Create parameter sharing if flag is True
        if share_weights:
            self.module3.weight = self.module1.weight


class NestedModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.submodule1 = nn.Linear(10, 20)
        self.submodule2 = nn.Linear(20, 10)


class RootModelWithNestedSharing(nn.Module):

    def __init__(self):
        super().__init__()
        self.nested1 = NestedModule()
        self.nested2 = NestedModule()
        # Tie weights between nested modules
        self.nested2.submodule1.weight = self.nested1.submodule1.weight


@_context
def test_no_tied_params():
    """Test when there are no tied parameters."""
    module1 = nn.Linear(10, 20)
    module2 = nn.Linear(20, 30)
    module3 = nn.Linear(30, 10)

    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module1, module2, module3])

    assert len(modules_to_shard) == 3
    assert len(modules_with_tied_params) == 0
    assert set(modules_to_shard) == {module1, module2, module3}


@_context
def test_with_tied_params_in_single_module():
    """Test when there are tied parameters within a single module."""
    module = ModuleWithTiedParams()

    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module])

    # The module has tied parameters internally, but there's only one module
    # so it should still be considered for sharding
    assert len(modules_to_shard) == 1
    assert len(modules_with_tied_params) == 0
    assert modules_to_shard[0] == module


@_context
def test_with_tied_params_across_modules():
    """Test when there are tied parameters across different modules."""
    model = MultiLinearTiedModel(share_weights=True)
    modules: list[nn.Module] = [model.module1, model.module2, model.module3]

    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules(modules)

    assert len(modules_to_shard) == 1
    assert modules_to_shard[0] == model.module2
    assert len(modules_with_tied_params) == 2
    assert model.module1 in modules_with_tied_params
    assert model.module3 in modules_with_tied_params


@_context
def test_empty_module_list():
    """Test with an empty list of modules."""
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([])

    assert len(modules_to_shard) == 0
    assert len(modules_with_tied_params) == 0


@_context
def test_modules_with_no_params():
    """Test with modules that have no parameters."""
    module1 = nn.ReLU()
    module2 = nn.ReLU()

    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module1, module2])

    assert len(modules_to_shard) == 0  # No modules with parameters
    assert len(modules_with_tied_params) == 0


@_context
def test_mixed_param_and_no_param_modules():
    """Test with a mix of modules with and without parameters."""
    module1 = nn.Linear(10, 10)
    module2 = nn.ReLU()
    module3 = nn.Linear(10, 10)

    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module1, module2, module3])

    assert len(modules_to_shard) == 2
    assert module1 in modules_to_shard
    assert module3 in modules_to_shard
    assert len(modules_with_tied_params) == 0


@_context
def test_mixed_tied_and_untied_modules():
    """Test with a mix of modules with and without tied parameters."""
    module1 = nn.Linear(10, 20)
    module2 = nn.Linear(20, 30)
    module3 = nn.Linear(10, 20)
    # Tie weights between module1 and module3
    module3.weight = module1.weight
    module4 = nn.Linear(30, 40)

    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module1, module2, module3, module4])

    assert len(modules_to_shard) == 2
    assert module2 in modules_to_shard
    assert module4 in modules_to_shard
    assert len(modules_with_tied_params) == 2
    assert module1 in modules_with_tied_params
    assert module3 in modules_with_tied_params


@_context
def test_tied_bias_only():
    """Test when only bias parameters are tied."""
    module1 = nn.Linear(10, 20)
    module2 = nn.Linear(10, 20)
    # Only tie bias parameters
    module2.bias = module1.bias

    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module1, module2])

    assert len(modules_to_shard) == 0
    assert len(modules_with_tied_params) == 2
    assert module1 in modules_with_tied_params
    assert module2 in modules_with_tied_params


@_context
def test_complex_nested_tied_params():
    """Test with complex nested modules with tied parameters."""
    nested1 = NestedModule()
    nested2 = NestedModule()
    # Tie a parameter between the nested modules
    nested2.submodule1.weight = nested1.submodule1.weight

    # We're testing the modules at the top level
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([nested1, nested2])

    # Both modules contain tied parameters
    assert len(modules_to_shard) == 0
    assert len(modules_with_tied_params) == 2
    assert nested1 in modules_with_tied_params
    assert nested2 in modules_with_tied_params


@_context
def test_legalize_param_sharing_no_sharing():
    """Test when there's no parameter sharing between modules_to_shard and other modules."""
    model = MultiLinearTiedModel(share_weights=False)
    # No sharing between these modules, so should pass without error
    modules_to_shard: list[nn.Module] = [model.module1, model.module3]

    # Should not raise an error
    legalize_param_sharing_between_modules(model, modules_to_shard)


@_context
def test_legalize_param_sharing_with_illegal_sharing():
    """Test when there's parameter sharing between modules_to_shard and other modules."""
    model = MultiLinearTiedModel(share_weights=True)
    # Only include module1 in modules_to_shard, not module3
    modules_to_shard: list[nn.Module] = [model.module1]

    # Should raise a ValueError
    with pytest.raises(ValueError):
        legalize_param_sharing_between_modules(model, modules_to_shard)


@_context
def test_legalize_param_sharing_with_nested_modules():
    """Test with nested modules and parameter sharing."""
    model = RootModelWithNestedSharing()
    # Only include nested1 but not nested2
    modules_to_shard: list[nn.Module] = [model.nested1]

    # Should raise a ValueError
    with pytest.raises(ValueError):
        legalize_param_sharing_between_modules(model, modules_to_shard)


@_context
def test_legalize_param_sharing_empty_modules_to_shard():
    """Test with an empty list of modules_to_shard."""
    model = MultiLinearTiedModel(share_weights=False)
    modules_to_shard: list[nn.Module] = []

    # Should not raise an error
    legalize_param_sharing_between_modules(model, modules_to_shard)


@_context
def test_legalize_param_sharing_all_modules_to_shard():
    """Test when all modules that share parameters are in modules_to_shard."""
    model = MultiLinearTiedModel(share_weights=True)
    # Include both module1 and module3 (which share parameters)
    modules_to_shard: list[nn.Module] = [model.module1, model.module3]

    # Should not raise an error
    legalize_param_sharing_between_modules(model, modules_to_shard)


@_context
def test_legalize_param_sharing_with_no_param_modules():
    """Test with modules that have no parameters."""

    class RootModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.module1 = nn.Linear(10, 20)
            self.module2 = nn.ReLU()  # No parameters

    model = RootModel()
    modules_to_shard: list[nn.Module] = [model.module1]

    # Should not raise an error
    legalize_param_sharing_between_modules(model, modules_to_shard)


@_context
def test_legalize_param_sharing_with_nested_shared_module():
    """Test where a module B is a submodule of both the root model and another module A."""

    class ModuleB(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 15)

    class ModuleA(nn.Module):

        def __init__(self, module_b):
            super().__init__()
            self.linear = nn.Linear(15, 20)
            self.shared_module = module_b  # B is a submodule of A

    class RootModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.module_b = ModuleB()  # B is a direct submodule of root
            self.module_a = ModuleA(self.module_b)  # A contains B

    model = RootModel()

    # Test 1: Include module_a but not module_b
    # Should raise error since module_b is used in module_a but not included
    modules_to_shard: list[nn.Module] = [model.module_a]
    with pytest.raises(ValueError):
        legalize_param_sharing_between_modules(model, modules_to_shard)

    # Test 2: Include both module_a and module_b
    # Should not raise an error
    modules_to_shard = [model.module_a, model.module_b]
    legalize_param_sharing_between_modules(model, modules_to_shard)

    # Test 3: Include only module_b
    # Should not raise an error
    modules_to_shard = [model.module_b]
    legalize_param_sharing_between_modules(model, modules_to_shard)

class ComprehensiveWrapModel(nn.Module):
    """A model combining various _fsdp_wrap scenarios."""

    def __init__(self):
        super().__init__()
        # block_a will be added since it has _fsdp_wrap set to True,
        # but block_a's child will not be added
        self.block_a = nn.Sequential(nn.Linear(10, 10))
        self.block_a._fsdp_wrap = True

        # block_b and it's children will not be added since block_b has _fsdp_wrap set to False
        self.block_b = nn.Sequential(nn.Linear(10, 10))
        self.block_b._fsdp_wrap = False
        self.block_b[0]._fsdp_wrap = True  # block_b_child

        # block_c won't be added since it has no _fsdp_wrap attribute
        # but block_c's children will be added
        self.block_c = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        )
        self.block_c[0]._fsdp_wrap = True
        self.block_c[1]._fsdp_wrap = True

        # block_d will be added since it has _fsdp_wrap set to True,
        # but block_d's child will not be added
        self.block_d = nn.Sequential(nn.Linear(10, 10))
        self.block_d._fsdp_wrap = True
        self.block_d[0]._fsdp_wrap = True

        # block_e won't be added since it has no _fsdp_wrap attribute
        self.block_e = nn.Linear(10, 10)

        # shared_block will be added since it has _fsdp_wrap set to True
        # but shared_block's parents will not be added
        self.shared_block = nn.Linear(10, 10)
        self.shared_block._fsdp_wrap = True
        self.shared_parent_1 = nn.Sequential(self.shared_block)
        self.shared_parent_2 = nn.Sequential(self.shared_block)

        # tied1 and tied2 will be added since they have _fsdp_wrap set to True
        # this is used for later tests
        self.tied1 = nn.Linear(10, 10)
        self.tied2 = nn.Linear(10, 10)
        self.untied = nn.Linear(10, 10)
        self.tied1._fsdp_wrap = True
        self.tied2._fsdp_wrap = True
        self.untied._fsdp_wrap = True
        self.tied2.weight = self.tied1.weight


@_context
def test_get_valid_modules_to_shard_comprehensive():
    """Tests get_valid_modules_to_shard with the comprehensive model."""
    model = ComprehensiveWrapModel()
    actual_modules = get_valid_modules_to_shard(model)

    # Define expected modules based on _fsdp_wrap rules
    expected_modules = {
        model.block_a,          # block_a has _fsdp_wrap set to True
        model.block_c[0],       # block_c has no _fsdp_wrap attribute, but block_c's children will be added
        model.block_c[1],       # block_c has no _fsdp_wrap attribute, but block_c's children will be added
        model.block_d,          # block_d has _fsdp_wrap set to True
        model.shared_block,     # shared_block has _fsdp_wrap set to True
        model.tied1,            # tied1 has _fsdp_wrap set to True
        model.tied2,            # tied2 has _fsdp_wrap set to True
        model.untied,           # untied has _fsdp_wrap set to True
    }

    assert set(actual_modules) == expected_modules, \
        f"Comprehensive test failed. Expected: {expected_modules}, Got: {set(actual_modules)}"


@_context
def test_interaction_tied_modules_selected_by_wrap():
    """Test interaction with tied weights using the comprehensive model."""
    model = ComprehensiveWrapModel()

    # Get modules based on _fsdp_wrap
    modules_from_wrap = get_valid_modules_to_shard(model)

    # Filter based on tied weights (simulating prepare_fully_shard)
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules(modules_from_wrap)

    # Expect only non-tied modules from the 'wrap' list
    # model.tied1 and model.tied2 are tied, so they won't be added, but
    # apply_fully_shard will apply the sharding to the parent model.
    expected_standalone = {
        model.block_a, model.block_c[0], model.block_c[1], model.block_d,
        model.shared_block, model.untied
    }
    expected_tied = {model.tied1, model.tied2}

    assert set(modules_to_shard) == expected_standalone
    assert modules_with_tied_params == expected_tied
