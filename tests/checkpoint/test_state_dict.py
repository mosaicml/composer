# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.checkpoint.state_dict import get_model_state_dict
from tests.common.models import EvenSimplerMLP, SimpleComposerMLP


@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_full(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cpu')
    else:
        model = EvenSimplerMLP(num_features=8, device='cpu')
    model_state_dict = get_model_state_dict(model, sharded=False, include_keys=None, ignore_keys=None)
    for name, param in model.named_parameters():
        print(name)
        assert name in model_state_dict
        assert torch.equal(model_state_dict[name], param)


@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_include(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cpu')
    else:
        model = EvenSimplerMLP(num_features=8, device='cpu')
    model_state_dict = get_model_state_dict(model, sharded=False, include_keys=['module.0.weight'])
    assert set(model_state_dict.keys()) == {'module.0.weight'}

    model_state_dict = get_model_state_dict(model, sharded=False, include_keys='module.2*')
    assert set(model_state_dict.keys()) == {'module.2.weight'}


@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_ignore(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cpu')
    else:
        model = EvenSimplerMLP(num_features=8, device='cpu')

    model_state_dict = get_model_state_dict(model, sharded=False, ignore_keys='module.2.weight')
    assert set(model_state_dict.keys()) == {'module.0.weight'}

    model_state_dict = get_model_state_dict(model, sharded=False, ignore_keys=['module.2*'])
    assert set(model_state_dict.keys()) == {'module.0.weight'}


#TODO add tests for sharded and for precision
