# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Dict, Type

import pytest
import torch

from composer.core.types import ModelParameters
from composer.optim.optimizer_hparams import (AdamHparams, AdamWHparams, DecoupledAdamWHparams, DecoupledSGDWHparams,
                                              OptimizerHparams, RAdamHparams, RMSPropHparams, SGDHparams, get_optimizer)
from composer.trainer.trainer_hparams import optimizer_registry

optimizer_constructors: Dict[Type[OptimizerHparams], OptimizerHparams] = {
    AdamHparams: AdamHparams(),
    RAdamHparams: RAdamHparams(),
    AdamWHparams: AdamWHparams(),
    DecoupledAdamWHparams: DecoupledAdamWHparams(),
    SGDHparams: SGDHparams(lr=0.001),
    DecoupledSGDWHparams: DecoupledSGDWHparams(lr=0.001),
    RMSPropHparams: RMSPropHparams(lr=0.001),
}


@pytest.fixture
def dummy_parameters() -> ModelParameters:
    sizes = [(5, 5), (10, 10)]
    parameters = [torch.rand(size) for size in sizes]
    return parameters


@pytest.mark.parametrize("optimizer_name", optimizer_registry.keys())
def test_optimizer_initialization(optimizer_name, dummy_parameters):

    # create the optimizer hparams object
    optimizer_cls: Type[OptimizerHparams] = optimizer_registry[optimizer_name]
    optimizer_hparams = optimizer_constructors[optimizer_cls]

    # create the optimizer object using the hparams
    optimizer = get_optimizer(dummy_parameters, optimizer_hparams)
    assert isinstance(optimizer, optimizer_hparams.optimizer_object)
