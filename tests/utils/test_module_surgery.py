# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import List, Mapping, Tuple, Type, cast
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from composer.algorithms.blurpool import BlurMaxPool2d
from composer.utils import module_surgery
from tests.common import SimpleModel


class RecursiveLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

        # submodule has modified out_features to prevent infinite recursion during test
        self.submodule = nn.Linear(in_features, out_features - 1)


class SimpleReplacementPolicy(nn.Module):
    """Bundle the model, replacement function, and validation into one class."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.pool = nn.MaxPool2d(kernel_size=3)

    @staticmethod
    def maybe_replace_linear(module: torch.nn.Module, module_index: int):
        del module_index  # unused
        if module.out_features in (10, 9) and not isinstance(module, RecursiveLinear):
            return RecursiveLinear(cast(int, module.in_features), cast(int, module.out_features))
        return None

    @staticmethod
    def replace_pool(module: torch.nn.Module, module_index: int):
        assert isinstance(module, nn.MaxPool2d)
        return BlurMaxPool2d.from_maxpool2d(module, module_index)

    def policy(self) -> Mapping[Type[torch.nn.Module], module_surgery.ReplacementFunction]:
        return {
            nn.Linear: self.maybe_replace_linear,
            nn.MaxPool2d: self.replace_pool,
        }

    def validate_replacements(self, recurse_on_replacements: bool):
        assert type(self.fc1) is nn.Linear
        assert type(self.fc2) is RecursiveLinear
        assert type(self.pool) is BlurMaxPool2d

        if recurse_on_replacements:
            assert type(self.fc2.submodule) is RecursiveLinear
            assert type(self.fc2.submodule.submodule) is nn.Linear
        else:
            assert type(self.fc2.submodule) is nn.Linear


class ModuleIdxReplacementPolicy(SimpleReplacementPolicy):
    """Test replacing only the first instance of a Linear layer."""

    @staticmethod
    def maybe_replace_linear(module: torch.nn.Module, module_index: int):
        if module_index == 0:
            return RecursiveLinear(cast(int, module.in_features), cast(int, module.out_features))
        return None

    def validate_replacements(self, recurse_on_replacements: bool):
        del recurse_on_replacements  # unused
        assert type(self.fc1) is RecursiveLinear
        assert type(self.fc2) is nn.Linear
        assert type(self.fc1.submodule) is nn.Linear


class NoOpReplacementPolicy(SimpleReplacementPolicy):

    def policy(self):
        return {nn.Conv2d: Mock(side_effect=AssertionError('test should not match on this layer'))}

    def validate_replacements(self, recurse_on_replacements: bool):
        del recurse_on_replacements  # unused
        assert type(self.fc1) is nn.Linear
        assert type(self.fc2) is nn.Linear


@pytest.mark.parametrize('recurse_on_replacements', [True, False])
@pytest.mark.parametrize('model_cls', [
    SimpleReplacementPolicy,
    ModuleIdxReplacementPolicy,
    NoOpReplacementPolicy,
])
def test_module_replacement(model_cls: Type[SimpleReplacementPolicy], recurse_on_replacements: bool):
    model = model_cls()
    module_surgery.replace_module_classes(
        model,
        optimizers=None,
        policies=model.policy(),
        recurse_on_replacements=recurse_on_replacements,
    )

    model.validate_replacements(recurse_on_replacements)


@pytest.mark.gpu
def test_module_replacement_gpu():
    model = SimpleReplacementPolicy()
    model = model.cuda()
    module_surgery.replace_module_classes(
        model,
        optimizers=None,
        policies=model.policy(),
        recurse_on_replacements=False,
    )

    model.validate_replacements(False)

    # Validate the model devices are correct
    for p in itertools.chain(model.parameters(), model.buffers()):
        assert p.device.type == 'cuda'


class _CopyLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.empty((out_features, in_features)))
        self.bias = None

    @staticmethod
    def from_linear(module: torch.nn.Module, module_index: int = -1):
        assert isinstance(module.in_features, int)
        assert isinstance(module.out_features, int)
        ret = _CopyLinear(in_features=module.in_features, out_features=module.out_features)
        with torch.no_grad():
            # new param object
            assert isinstance(module.weight, torch.Tensor)
            ret.weight.copy_(module.weight)
            ret.bias = module.bias  # same param object
        return ret


@pytest.fixture
def optimizer_surgery_state():
    num_channels = 1
    n_classes = 10
    model = SimpleModel(num_channels, n_classes)
    policy: Mapping[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {
        torch.nn.Linear: _CopyLinear.from_linear
    }
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    orig_linear_modules = [model.fc1, model.fc2]
    module_surgery.replace_module_classes(model, policies=policy, optimizers=opt)
    new_linear_modules = [model.fc1, model.fc2]
    return orig_linear_modules, new_linear_modules, opt


def test_optimizer_surgery_no_duplicate_params(optimizer_surgery_state: Tuple[List[torch.nn.Module],
                                                                              List[torch.nn.Module], Optimizer]):
    _, _, opt = optimizer_surgery_state
    params_list = opt.param_groups[0]['params']
    params_set = set(params_list)
    assert len(params_list) == len(params_set)


def _param_in_optimizer(param: torch.nn.parameter.Parameter, opt: torch.optim.Optimizer):
    return module_surgery._find_param_in_optimizer(param, opt) >= 0


def test_optimizer_surgery_removed_params_gone(optimizer_surgery_state: Tuple[List[torch.nn.Module],
                                                                              List[torch.nn.Module], Optimizer]):
    orig_linear_modules, _, opt = optimizer_surgery_state
    for module in orig_linear_modules:
        assert isinstance(module.weight, torch.nn.parameter.Parameter)
        assert not _param_in_optimizer(module.weight, opt)


def test_optimizer_surgery_new_params_present(optimizer_surgery_state: Tuple[List[torch.nn.Module],
                                                                             List[torch.nn.Module], Optimizer]):
    _, new_linear_modules, opt = optimizer_surgery_state
    for module in new_linear_modules:
        assert isinstance(module.weight, torch.nn.parameter.Parameter)
        assert _param_in_optimizer(module.weight, opt)
        assert isinstance(module.bias, torch.nn.parameter.Parameter)
        assert _param_in_optimizer(module.bias, opt)


def test_optimizer_surgery_params_not_removed_still_there(optimizer_surgery_state: Tuple[List[torch.nn.Module],
                                                                                         List[torch.nn.Module],
                                                                                         Optimizer]):
    orig_linear_modules, _, opt = optimizer_surgery_state
    for module in orig_linear_modules:
        assert isinstance(module.bias, torch.nn.parameter.Parameter)
        assert _param_in_optimizer(module.bias, opt)


class ParamTestModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(64, 64)


def test_update_params_in_optimizer():
    m1 = ParamTestModel()
    m2 = ParamTestModel()
    optimizer = torch.optim.Adam(m1.parameters(), lr=0.01)
    current_order = list(m2.parameters())
    module_surgery.update_params_in_optimizer(old_params=m1.parameters(),
                                              new_params=m2.parameters(),
                                              optimizers=optimizer)
    post_replacement_order = optimizer.param_groups[0]['params']
    for idx, value in enumerate(current_order):
        assert torch.all(value.eq(post_replacement_order[idx]))
