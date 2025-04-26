# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor

from composer.utils.parallelism import FSDP2Config
from tests.common import world_size
from tests.trainer.fsdp2_context import (
    _generate_default_policy,
    _recursive_apply_fully_shard,
    check_param_tying,
    fsdp2_context,
    prepare_fully_shard,
)


class NestedModule(nn.Module):
    """A nested module with a deep nested structure."""

    def __init__(self, parent_num):
        super().__init__()
        assert parent_num in [2, 5], f'Invalid parent_num: {parent_num}'
        if parent_num == 2:
            self.m3 = nn.Linear(10, 10)
            self.m4 = nn.Linear(10, 10)
        else:
            self.m6 = nn.Linear(10, 10)
            self.m7 = nn.Linear(10, 10)


class DeepNestedModel(nn.Module):
    """A model with a deep nested structure."""

    def __init__(self):
        super().__init__()
        self.m2 = NestedModule(parent_num=2)
        self.m5 = NestedModule(parent_num=5)


def check_not_dtensors(params: list[torch.nn.Parameter]):
    for param in params:
        assert not isinstance(param, DTensor), f'{param} should not be a DTensor'


def check_dtensors(params: list[torch.nn.Parameter]):
    for param in params:
        assert isinstance(param, DTensor), f'{param} should be a DTensor'


# Model hierarchy for tests:
# M1 (root, DeepNestedModel)
# ├── M2 (NestedModule)
# │   ├── M3 (Linear)
# │   └── M4 (Linear)
# └── M5 (NestedModule)
#     ├── M6 (Linear)
#     └── M7 (Linear)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_separate_modules(world_size: int):
    """Test FSDP wrapping applied to separate, non-overlapping modules (M2, M5)."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2._fsdp_wrap = True  # type: ignore
    m1.m5._fsdp_wrap = True  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    prepare_fully_shard(m1, opt, fsdp2_config)
    check_dtensors(list(m1.parameters()))


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_error_tied_siblings(world_size: int):
    """Test error when siblings (M3, M4) with tied weights are both marked for FSDP wrapping."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2.m4.weight = m1.m2.m3.weight
    m1.m2.m3._fsdp_wrap = True  # type: ignore
    m1.m2.m4._fsdp_wrap = True  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    with pytest.raises(ValueError, match='Detected tied parameters between modules designated for FSDP wrapping'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_error_tied_sibling_one_wrapped(world_size: int):
    """Test error when one module (M3) marked for FSDP wrap shares weights with a sibling (M4)."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2.m4.weight = m1.m2.m3.weight
    m1.m2.m3._fsdp_wrap = True  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    with pytest.raises(ValueError, match='Parameter sharing detected between modules to be sharded and module'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_ancestor_with_tied_children(world_size: int):
    """Test wrapping an ancestor (M2) whose children (M3, M4) have tied weights. Should succeed."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2._fsdp_wrap = True  # type: ignore
    m1.m2.m3.weight = m1.m2.m4.weight
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    prepare_fully_shard(m1, opt, fsdp2_config)
    check_dtensors(list(m1.parameters()))
    error_msg = 'm1.m2.m3.weight and m1.m2.m4.weight should be the same object'
    assert id(m1.m2.m3.weight) == id(m1.m2.m4.weight), error_msg


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_error_tied_across_branches_ancestor_wrap(world_size: int):
    """Test error when a wrapped module (M2) has a child (M3) tied to a module (M6) in another branch."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2._fsdp_wrap = True  # type: ignore
    m1.m2.m3.weight = m1.m5.m6.weight  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    with pytest.raises(ValueError, match='Parameter sharing detected between modules to be sharded and module'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_root_with_tied_descendants(world_size: int):
    """Test wrapping the root (M1) when descendants (M3, M6) have tied weights. Should succeed."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1._fsdp_wrap = True  # type: ignore
    m1.m2.m3.weight = m1.m5.m6.weight  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    prepare_fully_shard(m1, opt, fsdp2_config)
    check_dtensors(list(m1.parameters()))
    error_msg = 'm1.m2.m3.weight and m1.m5.m6.weight should be the same object'
    assert id(m1.m2.m3.weight) == id(m1.m5.m6.weight), error_msg  # type: ignore


@world_size(2)
@fsdp2_context
def test_fsdp_manual_policy_submodule_only(world_size: int):
    """Test manual policy application where only a submodule (M3) is wrapped."""
    m1 = DeepNestedModel()
    m1._fsdp_wrap = False  # type: ignore # Ensure root is not wrapped by default policy
    m1.m2._fsdp_wrap = False  # type: ignore # Ensure intermediate is not wrapped
    m1.m2.m3._fsdp_wrap = True  # type: ignore # Target module for wrapping
    auto_wrap_policy = _generate_default_policy(m1)
    target_modules_to_kwargs = auto_wrap_policy._run_policy(root_module=m1, ignored_modules=set(), root_kwargs={})
    _recursive_apply_fully_shard(m1, m1, target_modules_to_kwargs)
    # Check only m1.m2.m3 parameters are DTensors
    check_dtensors(list(m1.m2.m3.parameters()))
    other_params = [p for p in m1.parameters() if p not in set(m1.m2.m3.parameters())]
    check_not_dtensors(other_params)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_parent_shares_with_child_parent_wrap(world_size: int):
    """Test wrapping a parent (M2) that shares weights with its child (M3). Should succeed."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2._fsdp_wrap = True  # type: ignore
    m1.m2.weight = m1.m2.m3.weight  # type: ignore # Assign M3's weight to M2
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    prepare_fully_shard(m1, opt, fsdp2_config)
    check_dtensors(list(m1.parameters()))
    error_msg = 'm1.m2.weight and m1.m2.m3.weight should be the same object'
    # Need to re-fetch the module if FSDP replaces it
    m2_final = m1.get_submodule('m2')
    m3_final = m2_final.get_submodule('m3')
    assert hasattr(m2_final, 'weight'), "Wrapped M2 should still have a 'weight' attribute"
    assert id(m2_final.weight) == id(m3_final.weight), error_msg  # type: ignore


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_error_parent_child_share_both_wrap(world_size: int):
    """Test error when parent (M2) and child (M3) share weights and both are marked for wrapping."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2._fsdp_wrap = True  # type: ignore
    m1.m2.m3._fsdp_wrap = True  # type: ignore
    m1.m2.weight = m1.m2.m3.weight  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    # The error message might vary depending on traversal order, matching broader pattern
    with pytest.raises(ValueError, match='Parameter sharing detected'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_error_parent_child_share_child_wrap(world_size: int):
    """Test error when parent (M2) and child (M3) share weights and only the child is marked for wrapping."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2._fsdp_wrap = False  # type: ignore
    m1.m2.m3._fsdp_wrap = True  # type: ignore
    m1.m2.weight = m1.m2.m3.weight  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    with pytest.raises(ValueError, match='Parameter sharing detected between modules to be sharded and module'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_error_complex_sharing_parent_wrap(world_size: int):
    """Test error with complex sharing: parent (M2) shares with child (M3), child shares with other branch (M6), parent is wrapped."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2._fsdp_wrap = True  # type: ignore
    m1.m2.weight = m1.m2.m3.weight  # type: ignore
    m1.m2.m3.weight = m1.m5.m6.weight  # type: ignore # M3 and M6 also share weights
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    # M2 is wrapped, its param m2.weight is tied to m2.m3.weight, which is tied to m5.m6.weight (outside M2 FSDP unit)
    with pytest.raises(ValueError, match='Parameter sharing detected between modules to be sharded and module'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_error_tied_across_branches_one_wrap(world_size: int):
    """Test error (as noted in original comments) when weights are tied (M3, M6) but only one (M6) is marked for wrap."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2.m3._fsdp_wrap = False  # type: ignore
    m1.m5.m6._fsdp_wrap = True  # type: ignore
    m1.m2.m3.weight = m1.m5.m6.weight  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    # M6 is marked for wrap, but shares weight with M3 which is not.
    with pytest.raises(ValueError, match='Parameter sharing detected between modules to be sharded and module'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_fn_base(world_size: int):
    """Test that a custom wrap function can be provided to the FSDP2Config."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()

    def wrap_fn(module: nn.Module):
        return True

    m1.fsdp_wrap_fn = wrap_fn  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    prepare_fully_shard(m1, opt, fsdp2_config)
    # all parameters should be DTensors
    check_dtensors(list(m1.parameters()))


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_fn_invalid_keys(world_size: int):
    """Test that an error is raised if the wrap function returns a dict with invalid keys."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()

    def wrap_fn(module: nn.Module):
        return {'tacos': False}

    m1.fsdp_wrap_fn = wrap_fn  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    with pytest.raises(ValueError, match='Invalid FSDP2 config keys in wrap_fn return value. Valid keys are: {'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_fn_target_module(world_size: int):
    """Test that if root module is not wrapped, the wrap function will wrap the target module."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1._fsdp_wrap = False  # type: ignore

    def wrap_fn(module: nn.Module):
        return module == m1.m2

    m1.fsdp_wrap_fn = wrap_fn  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    prepare_fully_shard(m1, opt, fsdp2_config)
    # only m1.m2 should be wrapped
    check_dtensors(list(m1.m2.parameters()))
    other_params = [p for p in m1.parameters() if p not in set(m1.m2.parameters())]
    check_not_dtensors(other_params)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_fn_error_tied_siblings(world_size: int):
    """Test that if weights are tied and wrapped incorrectly, an error is raised."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()

    def wrap_fn(module: nn.Module):
        return module == m1.m2

    m1.fsdp_wrap_fn = wrap_fn  # type: ignore
    m1.m2.m3.weight = m1.m5.m6.weight
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    with pytest.raises(ValueError, match='Parameter sharing detected'):
        prepare_fully_shard(m1, opt, fsdp2_config)


@world_size(2)
@fsdp2_context
def test_fsdp_wrap_fn_reshard_after_forward(world_size: int):
    """Test if the wrap function can correctly return a dictionary of kwargs."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()

    def wrap_fn(module: nn.Module):
        return {'reshard_after_forward': False}

    m1.fsdp_wrap_fn = wrap_fn  # type: ignore
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)
    prepare_fully_shard(m1, opt, fsdp2_config)

    def check_reshard_after_forward(module: nn.Module):
        fsdp_state = module._get_fsdp_state()  # type: ignore
        param_group = fsdp_state._fsdp_param_group
        assert param_group.post_forward_mesh_info is None, \
            f'reshard_after_forward should be False, but got {param_group.post_forward_mesh_info}'

    # For this, we can only check leaf modules otherwise, we will run into errors
    for module in [m1.m2.m3, m1.m2.m4, m1.m5.m6, m1.m5.m7]:
        check_reshard_after_forward(module)


@fsdp2_context
@world_size(2)
def test_check_param_tying(world_size: int):
    """Test that if weights are tied different before and after the context, an error is raised."""
    m1 = DeepNestedModel()
    m1.m2.m3.weight = m1.m2.m4.weight

    def update_model(m1):
        m1.m2.m3.weight = m1.m5.m6.weight

    with pytest.raises(RuntimeError, match='Parameter tying relationship changed during the context'):
        with check_param_tying(m1):  # type: ignore
            update_model(m1)


@world_size(2)
@fsdp2_context
def test_check_param_tying_fsdp_wrap(world_size: int):
    """Test that if weights are tied different before and after the context, an error is raised."""
    fsdp2_config = FSDP2Config()
    m1 = DeepNestedModel()
    m1.m2.m3.weight = m1.m2.m4.weight
    opt = torch.optim.Adam(m1.parameters(), lr=0.01)

    def update_model(m1):
        prepare_fully_shard(m1, opt, fsdp2_config)
        m1.m2.m3.weight = m1.m5.m6.weight

    with pytest.raises(RuntimeError, match='Parameter tying relationship changed during the context'):
        with check_param_tying(m1):  # type: ignore
            update_model(m1)
