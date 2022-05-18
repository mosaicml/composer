from typing import Callable

import pytest

import composer.optim.optimizer_hparams
from composer.optim.optimizer_hparams import OptimizerHparams, optimizer_registry
from tests.common import assert_is_constructable_from_yaml, assert_registry_contains_entry, get_all_subclasses_in_module
from tests.common.models import SimpleModel

optimizer_hparam_classes = get_all_subclasses_in_module(composer.optim.optimizer_hparams, OptimizerHparams)


@pytest.mark.parametrize("optimizer_hparams_cls", optimizer_hparam_classes)
class TestOptimizerHparams:

    def test_optimizer_in_registry(self, optimizer_hparams_cls: Callable[..., OptimizerHparams]):
        assert_registry_contains_entry(optimizer_hparams_cls, optimizer_registry)

    def test_optimizer_is_constructable(self, optimizer_hparams_cls: Callable[..., OptimizerHparams]):
        optimizer_hparams = optimizer_hparams_cls(lr=0.001)
        assert isinstance(optimizer_hparams, OptimizerHparams)
        assert optimizer_hparams.optimizer_cls is not None
        model = SimpleModel()
        optimizer = optimizer_hparams.initialize_object(model.parameters())
        assert isinstance(optimizer, optimizer_hparams.optimizer_cls)

    def test_optimizer_is_constructable_from_hparams(self, optimizer_hparams_cls: Callable[..., OptimizerHparams]):
        assert optimizer_hparams_cls.optimizer_cls is not None
        assert_is_constructable_from_yaml(optimizer_hparams_cls, {'lr': 0.001}, expected=optimizer_hparams_cls)
