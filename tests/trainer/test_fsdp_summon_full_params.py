# Copyright 2025 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
import torch
import torch.distributed.fsdp
from torch.distributed._tensor import DTensor

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tests.common import (
    PartialWeightTiedModel,
    SimpleComposerMLP,
    NestedFSDPModel,
    world_size,
)
from tests.trainer.test_fsdp2 import create_trainer_with_model
from composer.distributed.shared_utils import get_summon_params_fn
from composer.distributed.fsdp2_utils import summon_full_params_fsdp2, _get_params_to_summon_fsdp2


def assert_right_fsdp_summon_params_fn(fsdp_version: int, fn: Callable):
    if fsdp_version == 1:
        assert fn == FSDP.summon_full_params
    else:
        assert fn == summon_full_params_fsdp2


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_version', [1, 2])
def test_summon_full_params(
    world_size: int,
    fsdp_version: int,
):
    """Test summon_full_params actually works with FSDP(1/2) models."""
    del world_size
    model = SimpleComposerMLP(num_features=10, device='cuda')
    model.add_fsdp_wrap_attribute_to_children()
    trainer = create_trainer_with_model(
        use_fsdp2=fsdp_version == 2,
        model=model,
    )

    def get_total_param_size(model: torch.nn.Module):
        total_size = 0
        for param in model.parameters():
            if hasattr(param, 'to_local'):
                param = param.to_local()  # type: ignore
            if param.data is not None:
                total_size += param.data.numel()
        return total_size

    distributed_param_size = get_total_param_size(model)

    summon_full_params = get_summon_params_fn(model)
    assert_right_fsdp_summon_params_fn(fsdp_version, summon_full_params)

    # Test with writeback=True
    with summon_full_params(model):
        local_param_size = get_total_param_size(model)

    assert local_param_size > 0, 'Local param size should be > 0'
    assert distributed_param_size > 0, 'Distributed param size should be > 0'
    assert local_param_size > distributed_param_size * 1.5, \
        f'Local param size {local_param_size} should be > 1.5x distributed param size {distributed_param_size}'

    trainer.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_version', [1, 2])
def test_summon_full_params_with_writeback(
    world_size: int,
    fsdp_version: int,
):
    """Test summon_full_params with actual FSDP models."""
    del world_size
    model = SimpleComposerMLP(num_features=10, device='cuda')
    model.add_fsdp_wrap_attribute_to_children()
    trainer = create_trainer_with_model(
        use_fsdp2=fsdp_version == 2,
        model=model,
    )

    original_local_tensors = {
        name: param.data.clone() for name, param in model.named_parameters()
    }

    summon_full_params = get_summon_params_fn(model)
    assert_right_fsdp_summon_params_fn(fsdp_version, summon_full_params)

    with summon_full_params(model, writeback=False):
        # Modify parameters inside the context
        for name, param in model.named_parameters():
            if param.data is not None:  # type: ignore
                param.data.fill_(777.0)

    for name, param in model.named_parameters():
        if param.data is not None:  # type: ignore
            assert torch.all(
                param.data == original_local_tensors[name],
            ), f'Parameter {name} should not be modified with writeback=False'

    # Test with writeback=True
    with summon_full_params(model, writeback=True):
        for name, param in model.named_parameters():
            if param.data is not None:  # type: ignore
                param.data.fill_(888.0)

    for name, param in model.named_parameters():
        if param.data is not None:  # type: ignore
            assert torch.all(
                param.data == 888.0,
            ), f'Parameter {name} should be modified with writeback=True'

    trainer.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_version', [1, 2])
def test_summon_full_params_tied_weights_with_writeback(
    world_size: int,
    fsdp_version: int,
):
    """Test summon_full_params with tied weights behavior verification."""
    del world_size
    model = PartialWeightTiedModel(num_features=2, device='cuda')
    model.add_fsdp_wrap_attribute_to_children()

    trainer = create_trainer_with_model(
        use_fsdp2=fsdp_version == 2,
        model=model,
    )

    from pprint import pprint
    pprint(list(model.modules()))

    summon_full_params = get_summon_params_fn(model)
    assert_right_fsdp_summon_params_fn(fsdp_version, summon_full_params)

    # fill the tied weights with 999.0
    model.module[0].net[0].weight.data.fill_(999.0)  # type: ignore
    # assert weight tying
    assert torch.all(
        model.module[0].net[-1].weight.data == 999.0,  # type: ignore
    )  # type: ignore

    with summon_full_params(model, writeback=False):
        error_msg = 'Tied weights should be the same tensor object inside context'
        first_weight = model.module[0].net[0].weight  # type: ignore
        last_weight = model.module[0].net[-1].weight  # type: ignore
        assert first_weight is last_weight, error_msg

        first_weight.data.fill_(777.0)
        error_msg = 'Tied weights should be consistent inside context'
        assert torch.all(last_weight.data == 777.0), error_msg

    first_weight_same = torch.all(
        model.module[0].net[0].weight.data == 999.0,  # type: ignore
    )
    last_weight_same = torch.all(
        model.module[0].net[-1].weight.data == 999.0,  # type: ignore
    )

    assert first_weight_same, 'First tied weight should be the same with writeback=False'
    assert last_weight_same, 'Second tied weight should be the same with writeback=False'

    # Test writeback=True
    with summon_full_params(model, writeback=True):
        first_weight = model.module[0].net[0].weight  # type: ignore
        last_weight = model.module[0].net[-1].weight  # type: ignore
        error_msg = 'Tied weights should be the same tensor object inside context'
        assert first_weight is last_weight, error_msg

        first_weight.data.fill_(888.0)

        error_msg = 'Tied weights should be consistent inside context'
        assert torch.all(last_weight.data == 888.0), error_msg

    first_weight_changed = torch.all(
        model.module[0].net[0].weight.data == 888.0,  # type: ignore
    )
    last_weight_changed = torch.all(
        model.module[0].net[-1].weight.data == 888.0,  # type: ignore
    )

    assert first_weight_changed, 'First tied weight should keep modified value with writeback=True'
    assert last_weight_changed, 'Second tied weight should keep modified value with writeback=True'

    trainer.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('fsdp_version', [1, 2])
def test_summon_full_params_recurse(
    world_size: int,
    fsdp_version: int,
):
    """Test summon_full_params with recurse=False parameter."""
    del world_size
    model = SimpleComposerMLP(num_features=10, device='cuda')
    model.add_fsdp_wrap_attribute_to_children()
    trainer = create_trainer_with_model(
        use_fsdp2=fsdp_version == 2,
        model=model,
    )

    summon_full_params = get_summon_params_fn(model)
    assert_right_fsdp_summon_params_fn(fsdp_version, summon_full_params)

    with summon_full_params(model, recurse=False):
        for name, param in model.named_parameters(recurse=False):
            assert param.data is not None  # type: ignore
            assert '.' not in name

    with summon_full_params(model, recurse=True):
        param_names = [
            name for name, _ in model.named_parameters(recurse=True)
        ]
        assert any('.' in name for name in param_names)

    trainer.close()


@pytest.mark.gpu
@world_size(2)
def test_get_params_to_summon_fsdp2(
    world_size: int,
):
    """Test _get_params_to_summon_fsdp2 function with nested FSDP structure."""
    del world_size

    model = NestedFSDPModel(num_features=2, device='cuda')
    model.add_fsdp_wrap_attribute_to_children()

    trainer = create_trainer_with_model(
        use_fsdp2=True,
        model=model,
    )

    dtensor_params_recurse = _get_params_to_summon_fsdp2(
        model.module,  # type: ignore
        recurse=True,
    )
    dtensor_params_no_recurse = _get_params_to_summon_fsdp2(
        model.module,  # type: ignore
        recurse=False,
    )

    # Assert all are DTensors
    for param in dtensor_params_recurse.values():
        assert isinstance(
            param,
            DTensor,
        ), f'Parameter {param.name} should be a DTensor'
    for param in dtensor_params_no_recurse.values():
        assert isinstance(
            param,
            DTensor,
        ), f'Parameter {param.name} should be a DTensor'

    assert len(dtensor_params_recurse) == 4, 'Should have 4 DTensors'
    for (
        name,
        param,
    ), value in zip(dtensor_params_recurse.items(), [1.0, 2.0, 3.0, 4.0]):
        assert torch.all(
            param.data == value,
        ), f'Parameter {name} should have value {value}'
    assert len(dtensor_params_no_recurse) == 2, 'Should have 2 DTensors'
    for (name,
         param), value in zip(dtensor_params_no_recurse.items(), [1.0, 3.0]):
        assert torch.all(
            param.data == value,
        ), f'Parameter {name} should have value {value}'

    trainer.close()
