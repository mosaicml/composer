# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F

from composer.algorithms import StochasticDepth, StochasticDepthHparams
from composer.algorithms.stochastic_depth.sample_stochastic_layers import SampleStochasticBottleneck
from composer.algorithms.stochastic_depth.stochastic_depth import STOCHASTIC_LAYER_MAPPING
from composer.algorithms.stochastic_depth.stochastic_layers import StochasticBottleneck, _sample_bernoulli
from composer.core import Event, Logger, State, surgery
from composer.core.types import Precision
from composer.datasets import SyntheticDatasetHparams
from composer.datasets.dataloader import DataloaderHparams
from composer.loggers import Logger
from composer.models import ResNet50Hparams
from tests.fixtures.dummy_fixtures import get_dataloader


@pytest.fixture()
def dummy_state(dummy_dataloader_hparams: DataloaderHparams):
    model = ResNet50Hparams(num_classes=100).initialize_object()
    dataset_hparams = SyntheticDatasetHparams(total_dataset_size=1000000,
                                              data_shape=[3, 32, 32],
                                              num_classes=10,
                                              device="cpu",
                                              drop_last=True,
                                              shuffle=False)
    train_dataloader = get_dataloader(dataset_hparams.initialize_object(), dummy_dataloader_hparams, batch_size=100)
    return State(epoch=50,
                 step=50,
                 train_dataloader=train_dataloader,
                 train_batch_size=100,
                 eval_batch_size=100,
                 grad_accum=1,
                 max_epochs=100,
                 model=model,
                 eval_dataloader=train_dataloader,
                 precision=Precision.FP32)


@pytest.mark.parametrize('stochastic_method', ['block', 'sample'])
@pytest.mark.parametrize('target_layer_name', ['ResNetBottleneck'])
def test_stochastic_depth_bottleneck_replacement(dummy_state, stochastic_method, target_layer_name, noop_dummy_logger):
    target_layer, stochastic_layer = STOCHASTIC_LAYER_MAPPING[stochastic_method][target_layer_name]
    target_block_count = surgery.count_module_instances(dummy_state.model, target_layer)

    sd = StochasticDepth(stochastic_method=stochastic_method,
                         target_layer_name=target_layer_name,
                         drop_rate=0.5,
                         drop_distribution='linear',
                         drop_warmup=0.0,
                         use_same_gpu_seed=False)
    sd.apply(Event.INIT, dummy_state, noop_dummy_logger)
    stochastic_block_count = surgery.count_module_instances(dummy_state.model, stochastic_layer)

    assert target_block_count == stochastic_block_count


@pytest.fixture
def device_ids():
    return torch.arange(8)


@pytest.fixture
def module_ids():
    return torch.arange(16)


@pytest.fixture
def probability():
    return 0.5


def test_sample_bernoulli_use_same_gpu_seed(probability, device_ids, module_ids):
    mask_matrix = torch.zeros(len(device_ids), len(module_ids))
    for device_id in device_ids:
        torch.manual_seed(0)  # Simulates each device having the same random_seed as is the case with DDP
        for module_id in module_ids:
            generator = torch.Generator().manual_seed(144385)
            mask_matrix[device_id, module_id] = _sample_bernoulli(probability=torch.tensor(probability),
                                                                  device_id=device_id.item(),
                                                                  module_id=module_id.item(),
                                                                  num_modules=len(module_ids),
                                                                  generator=generator,
                                                                  use_same_gpu_seed=True,
                                                                  use_same_depth_across_gpus=True)

    assert torch.unique(mask_matrix, dim=0).size(0) == 1


def test_sample_bernoulli_devices_not_same_gpu(probability, device_ids, module_ids):
    mask_matrix = torch.zeros(len(device_ids), len(module_ids))
    for device_id in device_ids:
        torch.manual_seed(0)  # Simulates each device having the same random_seed as is the case with DDP
        for module_id in module_ids:
            generator = torch.Generator().manual_seed(144385)
            mask_matrix[device_id, module_id] = _sample_bernoulli(probability=torch.tensor(probability),
                                                                  device_id=device_id.item(),
                                                                  module_id=module_id.item(),
                                                                  num_modules=len(module_ids),
                                                                  generator=generator,
                                                                  use_same_gpu_seed=False,
                                                                  use_same_depth_across_gpus=False)
    # Check for unique drop masks across devices
    assert (torch.unique(mask_matrix, dim=0).size(0) != 1)


def test_sample_bernoulli_layers_not_same_gpu(probability, device_ids, module_ids):
    mask_matrix = torch.zeros(len(device_ids), len(module_ids))
    for device_id in device_ids:
        torch.manual_seed(0)  # Simulates each device having the same random_seed as is the case with DDP
        for module_id in module_ids:
            generator = torch.Generator().manual_seed(144385)
            mask_matrix[device_id, module_id] = _sample_bernoulli(probability=torch.tensor(probability),
                                                                  device_id=device_id.item(),
                                                                  module_id=module_id.item(),
                                                                  num_modules=len(module_ids),
                                                                  generator=generator,
                                                                  use_same_gpu_seed=False,
                                                                  use_same_depth_across_gpus=False)
    # Check for unique drop masks across layers
    assert (torch.unique(mask_matrix, dim=1).size(1) != 1)


@pytest.mark.parametrize('drop_rate', [1.0])
def test_stochastic_bottleneck_drop(drop_rate):
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
def test_stochastic_bottleneck_keep(drop_rate):
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
def test_sample_stochastic_bottleneck_drop_all(drop_rate):
    x = F.relu(torch.randn(4, 4, 16, 16))  # inputs and outputs will match if the input has been ReLUed
    sample_stochastic_block = SampleStochasticBottleneck(drop_rate=drop_rate, inplanes=4, planes=1)
    x_dropped = sample_stochastic_block(x)
    assert torch.all(x == x_dropped)


@pytest.fixture()
def target_layer_name() -> str:
    return "ResNetBottleneck"


@pytest.fixture()
def stochastic_method():
    return "sample"


@pytest.fixture
def dummy_hparams(stochastic_method, target_layer_name, drop_rate, drop_distribution, drop_warmup, use_same_gpu_seed):
    return StochasticDepthHparams(stochastic_method, target_layer_name, drop_rate, drop_distribution, drop_warmup,
                                  use_same_gpu_seed)


def get_drop_rate_list(module: torch.nn.Module, drop_rates=[]):
    if (len(list(module.children())) == 0 and len(list(module.parameters())) > 0):
        return
    else:
        for name, child in module.named_children():
            if hasattr(child, 'drop_rate'):
                drop_rates.append(child.drop_rate)
            get_drop_rate_list(child, drop_rates)


@pytest.mark.parametrize("step", [50, 100, 1000])
@pytest.mark.parametrize("drop_rate", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("drop_distribution", ['uniform', 'linear'])
@pytest.mark.parametrize("use_same_gpu_seed", [True])
@pytest.mark.parametrize("drop_warmup", [0.1])
def test_drop_rate_warmup(step, dummy_hparams, dummy_state, noop_dummy_logger: Logger):
    dummy_algorithm = dummy_hparams.initialize_object()
    old_drop_rates = []
    get_drop_rate_list(dummy_state.model, drop_rates=old_drop_rates)
    dummy_state.step = step
    dummy_algorithm.apply(Event.BATCH_START, dummy_state, noop_dummy_logger)
    new_drop_rates = []
    get_drop_rate_list(dummy_state.model, drop_rates=new_drop_rates)

    drop_warmup_iters = dummy_state.steps_per_epoch * dummy_state.max_epochs * dummy_algorithm.hparams.drop_warmup
    assert torch.all(torch.tensor(new_drop_rates) == ((step / drop_warmup_iters) * torch.tensor(old_drop_rates)))


@pytest.mark.parametrize("stochastic_method", ['nonsense'])
def test_invalid_method_name(stochastic_method, target_layer_name):
    with pytest.raises(ValueError):
        sd_hparams = StochasticDepthHparams(stochastic_method=stochastic_method, target_layer_name=target_layer_name)
        sd_hparams.validate()


@pytest.mark.parametrize("target_layer_name", ['nonsense_pt2'])
def test_invalid_layer_name(stochastic_method, target_layer_name):
    with pytest.raises(ValueError):
        sd_hparams = StochasticDepthHparams(stochastic_method=stochastic_method, target_layer_name=target_layer_name)
        sd_hparams.validate()


@pytest.mark.parametrize("drop_rate", [-0.5, 1.7])
def test_invalid_drop_rate(stochastic_method, target_layer_name, drop_rate):
    with pytest.raises(ValueError):
        sd_hparams = StochasticDepthHparams(stochastic_method=stochastic_method,
                                            target_layer_name=target_layer_name,
                                            drop_rate=drop_rate)
        sd_hparams.validate()


@pytest.mark.parametrize("drop_distribution", ['nonsense_pt3'])
def test_invalid_drop_distribution(stochastic_method, target_layer_name, drop_distribution):
    with pytest.raises(ValueError):
        sd_hparams = StochasticDepthHparams(stochastic_method=stochastic_method,
                                            target_layer_name=target_layer_name,
                                            drop_distribution=drop_distribution)
        sd_hparams.validate()


@pytest.mark.parametrize("drop_warmup", [-0.5, 1.7])
def test_invalid_drop_warmup(stochastic_method, target_layer_name, drop_warmup):
    with pytest.raises(ValueError):
        sd_hparams = StochasticDepthHparams(stochastic_method,
                                            target_layer_name=target_layer_name,
                                            drop_warmup=drop_warmup)
        sd_hparams.validate()
