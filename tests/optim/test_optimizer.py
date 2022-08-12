# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Type

import pytest
import torch

from composer.optim.optimizer_hparams_registry import (AdamHparams, AdamWHparams, DecoupledAdamWHparams,
                                                       DecoupledSGDWHparams, OptimizerHparams, RAdamHparams,
                                                       RMSpropHparams, SGDHparams)
from composer.trainer.trainer_hparams import optimizer_registry

optimizer_constructors: Dict[Type[OptimizerHparams], OptimizerHparams] = {
    AdamHparams: AdamHparams(),
    RAdamHparams: RAdamHparams(),
    AdamWHparams: AdamWHparams(),
    DecoupledAdamWHparams: DecoupledAdamWHparams(),
    SGDHparams: SGDHparams(lr=0.001),
    DecoupledSGDWHparams: DecoupledSGDWHparams(lr=0.001),
    RMSpropHparams: RMSpropHparams(lr=0.001),
}


@pytest.fixture
def dummy_parameters():
    sizes = [(5, 5), (10, 10)]
    parameters = [torch.rand(size) for size in sizes]
    return parameters


@pytest.mark.parametrize('optimizer_name', optimizer_registry.keys())
def test_optimizer_initialization(optimizer_name, dummy_parameters):

    # create the optimizer hparams object
    optimizer_cls: Type[OptimizerHparams] = optimizer_registry[optimizer_name]
    optimizer_hparams = optimizer_constructors[optimizer_cls]

    # create the optimizer object using the hparams
    optimizer = optimizer_hparams.initialize_object(param_group=dummy_parameters)
    assert optimizer_hparams.optimizer_cls is not None
    assert isinstance(optimizer, optimizer_hparams.optimizer_cls)
