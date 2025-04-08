from packaging import version
from typing import Callable, Optional

import torch.nn as nn
import torch

if version.parse(torch.__version__) >= version.parse('2.6.0'):
    from composer.distributed.fsdp2 import get_standalone_and_tied_modules
    RUN_TEST = True
else:
    RUN_TEST = False
    get_standalone_and_tied_modules = lambda x: ([], set())


def test_context(func: Callable) -> Optional[Callable]:
    """Decorator to run tests with models initialized on the meta device for torch version 2.6+."""
    if not RUN_TEST:
        return None
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


class ComplexTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = nn.Linear(10, 20)
        self.module2 = nn.Linear(20, 10)
        self.module3 = nn.Linear(10, 20)
        # Tie weights between module1 and module3
        self.module3.weight = self.module1.weight
    
    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        return x


@test_context
def test_no_tied_params():
    """Test when there are no tied parameters."""
    module1 = nn.Linear(10, 20)
    module2 = nn.Linear(20, 30)
    module3 = nn.Linear(30, 10)
    
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module1, module2, module3])
    
    assert len(modules_to_shard) == 3
    assert len(modules_with_tied_params) == 0
    assert set(modules_to_shard) == {module1, module2, module3}


@test_context
def test_with_tied_params_in_single_module():
    """Test when there are tied parameters within a single module."""
    module = ModuleWithTiedParams()
    
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module])
    
    # The module has tied parameters internally, but there's only one module
    # so it should still be considered for sharding
    assert len(modules_to_shard) == 1
    assert len(modules_with_tied_params) == 0
    assert modules_to_shard[0] == module


@test_context
def test_with_tied_params_across_modules():
    """Test when there are tied parameters across different modules."""
    model = ComplexTiedModel()
    modules: list[nn.Module] = [model.module1, model.module2, model.module3]
    
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules(modules)
    
    assert len(modules_to_shard) == 1
    assert modules_to_shard[0] == model.module2
    assert len(modules_with_tied_params) == 2
    assert model.module1 in modules_with_tied_params
    assert model.module3 in modules_with_tied_params


@test_context
def test_empty_module_list():
    """Test with an empty list of modules."""
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([])
    
    assert len(modules_to_shard) == 0
    assert len(modules_with_tied_params) == 0


@test_context
def test_modules_with_no_params():
    """Test with modules that have no parameters."""
    module1 = nn.ReLU()
    module2 = nn.ReLU()
    
    modules_to_shard, modules_with_tied_params = get_standalone_and_tied_modules([module1, module2])
    
    assert len(modules_to_shard) == 0  # No modules with parameters
    assert len(modules_with_tied_params) == 0


@test_context
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


@test_context
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


@test_context
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


@test_context
def test_complex_nested_tied_params():
    """Test with complex nested modules with tied parameters."""
    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule1 = nn.Linear(10, 20)
            self.submodule2 = nn.Linear(20, 10)
    
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
