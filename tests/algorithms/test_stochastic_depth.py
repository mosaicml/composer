# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Type
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from composer.algorithms import StochasticDepth
from composer.algorithms.stochastic_depth.stochastic_depth import _STOCHASTIC_LAYER_MAPPING, apply_stochastic_depth
from composer.algorithms.stochastic_depth.stochastic_layers import make_resnet_bottleneck_stochastic
from composer.core import Event, State
from composer.core.time import TimeUnit
from composer.models import composer_resnet
from composer.utils import module_surgery


@pytest.fixture()
def state(minimal_state: State):
    """stochastic depth tests require ResNet model."""
    minimal_state.model = composer_resnet(model_name='resnet50', num_classes=100)
    return minimal_state


@pytest.fixture()
def target_layer_name() -> str:
    return 'ResNetBottleneck'


@pytest.fixture()
def stochastic_method():
    return 'block'


def count_sd_forward(module: torch.nn.Module, target_block: Type[torch.nn.Module], count: int = 0):
    if (len(list(module.children()))) == 0 and len(list(module.parameters())) > 0:
        return count
    else:
        for child in module.children():
            if isinstance(child, target_block) and hasattr(child, 'drop_rate'):
                count += 1
            count = count_sd_forward(child, target_block, count)
    return count


@pytest.mark.parametrize('stochastic_method', ['block', 'sample'])
@pytest.mark.parametrize('target_layer_name', ['ResNetBottleneck'])
def test_sd_algorithm(state: State, stochastic_method: str, target_layer_name: str):
    target_layer, _ = _STOCHASTIC_LAYER_MAPPING[target_layer_name]
    target_block_count = module_surgery.count_module_instances(state.model, target_layer)

    sd = StochasticDepth(stochastic_method=stochastic_method,
                         target_layer_name=target_layer_name,
                         drop_rate=0.5,
                         drop_distribution='linear',
                         drop_warmup=0.0)
    sd.apply(Event.INIT, state, logger=Mock())
    stochastic_forward_count = count_sd_forward(state.model, target_layer)

    assert target_block_count == stochastic_forward_count


@pytest.mark.parametrize('stochastic_method', ['block', 'sample'])
@pytest.mark.parametrize('target_layer_name', ['ResNetBottleneck'])
def test_sd_functional(state: State, stochastic_method: str, target_layer_name: str):
    target_layer, _ = _STOCHASTIC_LAYER_MAPPING[target_layer_name]
    target_block_count = module_surgery.count_module_instances(state.model, target_layer)

    apply_stochastic_depth(model=state.model,
                           stochastic_method=stochastic_method,
                           target_layer_name=target_layer_name,
                           drop_rate=0.5,
                           drop_distribution='linear')

    stochastic_forward_count = count_sd_forward(state.model, target_layer)

    assert target_block_count == stochastic_forward_count


class TestStochasticBottleneckForward:

    @pytest.mark.parametrize('drop_rate', [1.0])
    def test_block_stochastic_bottleneck_drop(self, drop_rate: float):
        X = torch.randn(4, 4, 16, 16)
        bottleneck_block = Bottleneck(inplanes=4, planes=1)
        stochastic_block = make_resnet_bottleneck_stochastic(module=bottleneck_block,
                                                             module_index=0,
                                                             module_count=1,
                                                             drop_rate=drop_rate,
                                                             drop_distribution='linear',
                                                             stochastic_method='block')
        stochastic_X = stochastic_block(X)
        assert stochastic_X is X

    @pytest.mark.parametrize('drop_rate', [0.0])
    def test_block_stochastic_bottleneck_keep(self, drop_rate: float):
        X = torch.randn(4, 4, 16, 16)
        bottleneck_block = Bottleneck(inplanes=4, planes=1)
        stochastic_block = make_resnet_bottleneck_stochastic(module=bottleneck_block,
                                                             module_index=0,
                                                             module_count=1,
                                                             drop_rate=drop_rate,
                                                             drop_distribution='linear',
                                                             stochastic_method='block')
        stochastic_X = stochastic_block(X)
        assert stochastic_X is not X

    @pytest.mark.parametrize('drop_rate', [1.0])
    def test_sample_stochastic_bottleneck_drop_all(self, drop_rate: float):
        X = F.relu(torch.randn(4, 4, 16, 16))  # inputs and outputs will match if the input has been ReLUed
        bottleneck_block = Bottleneck(inplanes=4, planes=1)
        stochastic_block = make_resnet_bottleneck_stochastic(module=bottleneck_block,
                                                             module_index=0,
                                                             module_count=1,
                                                             drop_rate=drop_rate,
                                                             drop_distribution='linear',
                                                             stochastic_method='sample')
        stochastic_X = stochastic_block(X)
        assert torch.all(X == stochastic_X)


class TestStochasticDepthDropRate:

    @pytest.fixture
    def algorithm(
        self,
        target_layer_name: str,
        stochastic_method: str,
        drop_rate: float,
        drop_distribution: str,
        drop_warmup: str,
    ):
        return StochasticDepth(
            target_layer_name,
            stochastic_method,
            drop_rate,
            drop_distribution,
            drop_warmup,
        )

    def get_drop_rate_list(self, module: torch.nn.Module, drop_rates: Optional[List] = None):
        if drop_rates is None:
            drop_rates = []
        if (len(list(module.children())) == 0 and len(list(module.parameters())) > 0):
            return
        else:
            for _, child in module.named_children():
                if hasattr(child, 'drop_rate'):
                    drop_rates.append(child.drop_rate)
                self.get_drop_rate_list(child, drop_rates)

    @pytest.mark.parametrize('step', [50, 100, 1000])
    @pytest.mark.parametrize('drop_rate', [0.0, 0.5, 1.0])
    @pytest.mark.parametrize('drop_distribution', ['uniform', 'linear'])
    @pytest.mark.parametrize('drop_warmup', ['0.1dur'])
    def test_drop_rate_warmup(self, algorithm: StochasticDepth, step: int, state: State):
        old_drop_rates = []
        self.get_drop_rate_list(state.model, drop_rates=old_drop_rates)
        state.timestamp._batch._value = step
        algorithm.apply(Event.BATCH_START, state, logger=Mock())
        new_drop_rates = []
        self.get_drop_rate_list(state.model, drop_rates=new_drop_rates)

        assert state.max_duration is not None
        assert state.max_duration.unit == TimeUnit.EPOCH
        assert state.dataloader_len is not None
        drop_warmup_iters = int(int(state.dataloader_len) * int(state.max_duration.value) * algorithm.drop_warmup)
        assert torch.all(torch.tensor(new_drop_rates) == ((step / drop_warmup_iters) * torch.tensor(old_drop_rates)))


class TestStochasticDepthInputValidation():

    @pytest.mark.parametrize('stochastic_method', ['nonsense'])
    def test_invalid_method_name(self, stochastic_method: str, target_layer_name: str):
        with pytest.raises(ValueError):
            StochasticDepth(stochastic_method=stochastic_method, target_layer_name=target_layer_name)

    @pytest.mark.parametrize('target_layer_name', ['nonsense_pt2'])
    def test_invalid_layer_name(self, stochastic_method: str, target_layer_name: str):
        with pytest.raises(ValueError):
            StochasticDepth(stochastic_method=stochastic_method, target_layer_name=target_layer_name)

    @pytest.mark.parametrize('drop_rate', [-0.5, 1.7])
    def test_invalid_drop_rate(self, stochastic_method: str, target_layer_name: str, drop_rate: float):
        with pytest.raises(ValueError):
            StochasticDepth(
                stochastic_method=stochastic_method,
                target_layer_name=target_layer_name,
                drop_rate=drop_rate,
            )

    @pytest.mark.parametrize('drop_distribution', ['nonsense_pt3'])
    def test_invalid_drop_distribution(self, stochastic_method: str, target_layer_name: str, drop_distribution: str):
        with pytest.raises(ValueError):
            StochasticDepth(stochastic_method=stochastic_method,
                            target_layer_name=target_layer_name,
                            drop_distribution=drop_distribution)
