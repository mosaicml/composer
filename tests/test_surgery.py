# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from composer.core import surgery


class RecursiveLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

        # submodule has modified out_features to prevent infinite recursion during test
        self.submodule = nn.Linear(in_features, out_features - 1)


class SimpleReplacementPolicy(nn.Module):
    """
    Bundle the model, replacement function, and validation into one class
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    @staticmethod
    def maybe_replace_linear(module, module_index):
        if module.out_features in (10, 9):
            return RecursiveLinear(module.in_features, module.out_features)
        return None

    def policy(self):
        return {nn.Linear: self.maybe_replace_linear}

    def validate_replacements(self, recurse_on_replacements):
        assert type(self.fc1) is nn.Linear
        assert type(self.fc2) is RecursiveLinear

        if recurse_on_replacements:
            assert type(self.fc2.submodule) is RecursiveLinear
            assert type(self.fc2.submodule.submodule) is nn.Linear
        else:
            assert type(self.fc2.submodule) is nn.Linear


class ModuleIdxReplacementPolicy(SimpleReplacementPolicy):
    """
    Test replacing only the first instance of a Linear layer
    """

    @staticmethod
    def maybe_replace_linear(module, module_index):
        if module_index == 0:
            return RecursiveLinear(module.in_features, module.out_features)
        return None

    def validate_replacements(self, recurse_on_replacements):
        assert type(self.fc1) is RecursiveLinear
        assert type(self.fc2) is nn.Linear
        assert type(self.fc1.submodule) is nn.Linear


class NoOpReplacementPolicy(SimpleReplacementPolicy):

    def policy(self):
        return {nn.Conv2d: Mock(side_effect=AssertionError('test should not match on this layer'))}

    def validate_replacements(self, recurse_on_replacements):
        assert type(self.fc1) is nn.Linear
        assert type(self.fc2) is nn.Linear


@pytest.mark.parametrize('recurse_on_replacements', [True, False])
@pytest.mark.parametrize('model', [
    SimpleReplacementPolicy,
    ModuleIdxReplacementPolicy,
    NoOpReplacementPolicy,
])
def test_module_replacement(model, recurse_on_replacements):
    model = model()
    surgery.replace_module_classes(
        model,
        model.policy(),
        recurse_on_replacements=recurse_on_replacements,
    )

    model.validate_replacements(recurse_on_replacements)


class _CopyLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = None

    @staticmethod
    def from_linear(module: torch.nn.Module, module_index: int = -1):
        ret = _CopyLinear(in_features=module.in_features, out_features=module.out_features)  # type: ignore
        with torch.no_grad():
            # new param object
            ret.weight.copy_(module.weight)  # type: ignore
            ret.bias = module.bias  # same param object
        return ret
