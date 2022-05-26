# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest

import composer.optim.optimizer_hparams
from composer.optim.optimizer_hparams import OptimizerHparams, optimizer_registry
from tests.common import get_module_subclasses
from tests.common.hparams import assert_in_registry, assert_yaml_loads
from tests.common.models import SimpleModel

optimizer_hparam_classes = get_module_subclasses(composer.optim.optimizer_hparams, OptimizerHparams)


@pytest.mark.parametrize("optimizer_hparams_cls", optimizer_hparam_classes)
class TestOptimizerHparams:

    def test_optimizer_in_registry(self, optimizer_hparams_cls: Callable[..., OptimizerHparams]):
        assert_in_registry(optimizer_hparams_cls, optimizer_registry)

    def test_optimizer_is_constructable(self, optimizer_hparams_cls: Callable[..., OptimizerHparams]):
        optimizer_hparams = optimizer_hparams_cls(lr=0.001)
        assert isinstance(optimizer_hparams, OptimizerHparams)
        assert optimizer_hparams.optimizer_cls is not None
        model = SimpleModel()
        optimizer = optimizer_hparams.initialize_object(model.parameters())
        assert isinstance(optimizer, optimizer_hparams.optimizer_cls)

    def test_optimizer_is_constructable_from_hparams(self, optimizer_hparams_cls: Callable[..., OptimizerHparams]):
        assert optimizer_hparams_cls.optimizer_cls is not None
        assert_yaml_loads(optimizer_hparams_cls, {'lr': 0.001}, expected=optimizer_hparams_cls)
