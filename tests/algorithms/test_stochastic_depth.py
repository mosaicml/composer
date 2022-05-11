# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from composer.algorithms import StochasticDepth, StochasticDepthHparams
from composer.algorithms.stochastic_depth.sample_stochastic_layers import SampleStochasticBottleneck
from composer.algorithms.stochastic_depth.stochastic_depth import _STOCHASTIC_LAYER_MAPPING, apply_stochastic_depth
from composer.algorithms.stochastic_depth.stochastic_layers import StochasticBottleneck, _sample_bernoulli
from composer.core import Event, State
from composer.core.time import TimeUnit
from composer.models import ComposerResNet
from composer.utils import module_surgery


@pytest.fixture()
def state(minimal_state: State):
    """stochastic depth tests require ResNet model."""
    minimal_state.model = ComposerResNet(model_name='resnet50', num_classes=100)
    return minimal_state


@pytest.fixture()
def target_layer_name() -> str:
    return "ResNetBottleneck"


@pytest.fixture()
def stochastic_method():
    return "block"


@pytest.mark.parametrize('stochastic_method', ['block', 'sample'])
@pytest.mark.parametrize('target_layer_name', ['ResNetBottleneck'])
def test_se_algorithm(state: State, stochastic_method: str, target_layer_name: str):
    target_layer, stochastic_layer = _STOCHASTIC_LAYER_MAPPING[stochastic_method][target_layer_name]
    target_block_count = module_surgery.count_module_instances(state.model, target_layer)

    sd = StochasticDepth(stochastic_method=stochastic_method,
                         target_layer_name=target_layer_name,
                         drop_rate=0.5,
                         drop_distribution='linear',
                         drop_warmup=0.0,
                         use_same_gpu_seed=False)
    sd.apply(Event.INIT, state, logger=Mock())
    stochastic_block_count = module_surgery.count_module_instances(state.model, stochastic_layer)

    assert target_block_count == stochastic_block_count


@pytest.mark.parametrize('stochastic_method', ['block', 'sample'])
@pytest.mark.parametrize('target_layer_name', ['ResNetBottleneck'])
def test_se_functional(state: State, stochastic_method: str, target_layer_name: str):
    target_layer, stochastic_layer = _STOCHASTIC_LAYER_MAPPING[stochastic_method][target_layer_name]
    target_block_count = module_surgery.count_module_instances(state.model, target_layer)

    apply_stochastic_depth(model=state.model,
                           stochastic_method=stochastic_method,
                           target_layer_name=target_layer_name,
                           drop_rate=0.5,
                           drop_distribution='linear',
                           use_same_gpu_seed=False)

    stochastic_block_count = module_surgery.count_module_instances(state.model, stochastic_layer)

    assert target_block_count == stochastic_block_count


def test_stochastic_depth_hparams(stochastic_method: str, target_layer_name: str):
    hparams = StochasticDepthHparams(
        stochastic_method=stochastic_method,
        target_layer_name=target_layer_name,
    )
    algorithm = hparams.initialize_object()
    assert isinstance(algorithm, StochasticDepth)


class TestSampleBernoulli:

    @pytest.fixture
    def device_ids(self):
        return torch.arange(8)

    @pytest.fixture
    def module_ids(self):
        return torch.arange(16)

    @pytest.fixture
    def probability(self):
        return 0.5

    @pytest.fixture
    def test_sample_bernoulli_use_same_gpu_seed(self, probability: float, device_ids: torch.Tensor,
                                                module_ids: torch.Tensor):
        mask_matrix = torch.zeros(len(device_ids), len(module_ids))
        for device_id in device_ids:
            torch.manual_seed(0)  # Simulates each device having the same random_seed as is the case with DDP
            for module_id in module_ids:
                generator = torch.Generator().manual_seed(144385)
                mask_matrix[device_id, module_id] = _sample_bernoulli(probability=torch.tensor(probability),
                                                                      device_id=int(device_id.item()),
                                                                      module_id=int(module_id.item()),
                                                                      num_modules=len(module_ids),
                                                                      generator=generator,
                                                                      use_same_gpu_seed=True,
                                                                      use_same_depth_across_gpus=True)

        assert torch.unique(mask_matrix, dim=0).size(0) == 1

    def test_sample_bernoulli_devices_not_same_gpu(self, probability: float, device_ids: Sequence[torch.Tensor],
                                                   module_ids: Sequence[torch.Tensor]):
        mask_matrix = torch.zeros(len(device_ids), len(module_ids))
        for device_id in device_ids:
            torch.manual_seed(0)  # Simulates each device having the same random_seed as is the case with DDP
            for module_id in module_ids:
                generator = torch.Generator().manual_seed(144385)
                mask_matrix[device_id, module_id] = _sample_bernoulli(probability=torch.tensor(probability),
                                                                      device_id=int(device_id.item()),
                                                                      module_id=int(module_id.item()),
                                                                      num_modules=len(module_ids),
                                                                      generator=generator,
                                                                      use_same_gpu_seed=False,
                                                                      use_same_depth_across_gpus=False)
        # Check for unique drop masks across devices
        assert (torch.unique(mask_matrix, dim=0).size(0) != 1)

    def test_sample_bernoulli_layers_not_same_gpu(self, probability: float, device_ids: torch.Tensor,
                                                  module_ids: torch.Tensor):
        mask_matrix = torch.zeros(len(device_ids), len(module_ids))
        for device_id in device_ids:
            torch.manual_seed(0)  # Simulates each device having the same random_seed as is the case with DDP
            for module_id in module_ids:
                generator = torch.Generator().manual_seed(144385)
                mask_matrix[device_id, module_id] = _sample_bernoulli(probability=torch.tensor(probability),
                                                                      device_id=int(device_id.item()),
                                                                      module_id=int(module_id.item()),
                                                                      num_modules=len(module_ids),
                                                                      generator=generator,
                                                                      use_same_gpu_seed=False,
                                                                      use_same_depth_across_gpus=False)
        # Check for unique drop masks across layers
        assert (torch.unique(mask_matrix, dim=1).size(1) != 1)


class TestStochasticBottleneckLayer:

    @pytest.mark.parametrize('drop_rate', [1.0])
    def test_stochastic_bottleneck_drop(self, drop_rate: float):
        X = torch.randn(4, 4, 16, 16)
        generator = torch.Generator().manual_seed(144385)
        stochastic_layer = StochasticBottleneck(drop_rate=drop_rate,
                                                module_id=2,
                                                module_count=10,
                                                use_same_depth_across_gpus=False,
                                                use_same_gpu_seed=False,
                                                rand_generator=generator,
                                                inplanes=4,
                                                planes=1)
        new_X = stochastic_layer(X)
        assert new_X is X

    @pytest.mark.parametrize('drop_rate', [0.0])
    def test_stochastic_bottleneck_keep(self, drop_rate: float):
        X = torch.randn(4, 4, 16, 16)
        generator = torch.Generator().manual_seed(144385)
        stochastic_layer = StochasticBottleneck(drop_rate=drop_rate,
                                                module_id=2,
                                                module_count=10,
                                                use_same_depth_across_gpus=False,
                                                use_same_gpu_seed=False,
                                                rand_generator=generator,
                                                inplanes=4,
                                                planes=1)
        new_X = stochastic_layer(X)
        assert new_X is not X

    @pytest.mark.parametrize('drop_rate', [1.0])
    def test_sample_stochastic_bottleneck_drop_all(self, drop_rate: float):
        x = F.relu(torch.randn(4, 4, 16, 16))  # inputs and outputs will match if the input has been ReLUed
        sample_stochastic_block = SampleStochasticBottleneck(drop_rate=drop_rate, inplanes=4, planes=1)
        x_dropped = sample_stochastic_block(x)
        assert torch.all(x == x_dropped)


class TestStochasticDepthDropRate:

    @pytest.fixture
    def algorithm(
        self,
        target_layer_name: str,
        stochastic_method: str,
        drop_rate: float,
        drop_distribution: str,
        drop_warmup: str,
        use_same_gpu_seed: bool,
    ):
        return StochasticDepth(
            target_layer_name,
            stochastic_method,
            drop_rate,
            drop_distribution,
            drop_warmup,
            use_same_gpu_seed,
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

    @pytest.mark.parametrize("step", [50, 100, 1000])
    @pytest.mark.parametrize("drop_rate", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("drop_distribution", ['uniform', 'linear'])
    @pytest.mark.parametrize("use_same_gpu_seed", [True])
    @pytest.mark.parametrize("drop_warmup", ["0.1dur"])
    @pytest.mark.timeout(5)
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

    @pytest.mark.parametrize("stochastic_method", ['nonsense'])
    def test_invalid_method_name(self, stochastic_method: str, target_layer_name: str):
        with pytest.raises(ValueError):
            StochasticDepth(stochastic_method=stochastic_method, target_layer_name=target_layer_name)

    @pytest.mark.parametrize("target_layer_name", ['nonsense_pt2'])
    def test_invalid_layer_name(self, stochastic_method: str, target_layer_name: str):
        with pytest.raises(ValueError):
            StochasticDepth(stochastic_method=stochastic_method, target_layer_name=target_layer_name)

    @pytest.mark.parametrize("drop_rate", [-0.5, 1.7])
    def test_invalid_drop_rate(self, stochastic_method: str, target_layer_name: str, drop_rate: float):
        with pytest.raises(ValueError):
            StochasticDepth(
                stochastic_method=stochastic_method,
                target_layer_name=target_layer_name,
                drop_rate=drop_rate,
            )

    @pytest.mark.parametrize("drop_distribution", ['nonsense_pt3'])
    def test_invalid_drop_distribution(self, stochastic_method: str, target_layer_name: str, drop_distribution: str):
        with pytest.raises(ValueError):
            StochasticDepth(stochastic_method=stochastic_method,
                            target_layer_name=target_layer_name,
                            drop_distribution=drop_distribution)
