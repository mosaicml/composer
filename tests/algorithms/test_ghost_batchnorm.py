# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Test Ghost Batch Normalization, both as an algorithm and module."""

import contextlib
import math
from typing import Any, Dict, Optional, Sequence, Union, cast
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torchmetrics import Metric

from composer.algorithms import ghost_batchnorm as ghostbn
from composer.algorithms.ghost_batchnorm.ghost_batchnorm import GhostBatchNorm, _GhostBatchNorm
from composer.core import Batch, Event, State
from composer.models import ComposerModel
from composer.utils import module_surgery

_GHOSTBN_MODULE_CLASS = _GhostBatchNorm
_GHOSTBN_CORRECT_EVENT = Event.INIT

_TEST_NUM_DIMS = [1, 2, 3]
_TEST_GHOST_BATCH_SIZES = [1, 2, 3, 5]
_TEST_BATCH_SIZES = [12]  # multiple of some, but not all, ghost batch sizes


class ModuleWithBatchnorm(ComposerModel):

    def __init__(self, num_dims, num_features=4):
        super().__init__()
        eps = 0  # makes it easier to check normalization correctness
        factory_func = {
            1: torch.nn.BatchNorm1d,
            2: torch.nn.BatchNorm2d,
            3: torch.nn.BatchNorm3d,
        }
        self.bn = factory_func[num_dims](num_features, eps=eps)
        self.num_dims = num_dims
        self.num_features = num_features
        self.non_batchnorm_module = torch.nn.Conv2d(4, 5, (1, 1))

    def forward(self, input: torch.Tensor):
        return self.bn(input)

    def loss(self, outputs: Any, batch: Batch, *args, **kwargs) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        raise NotImplementedError()

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        raise NotImplementedError()

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        raise NotImplementedError()


@pytest.fixture
def state(num_dims: int) -> State:
    return MagicMock(model=ModuleWithBatchnorm(num_dims=num_dims))


@pytest.fixture
def algo_instance(ghost_batch_size: int):
    return GhostBatchNorm(ghost_batch_size=ghost_batch_size)


@pytest.mark.parametrize('num_dims', [1, 2, 3, 4, -1])
def test_batchnorm_gets_replaced_functional(num_dims: int):
    if num_dims < 1 or num_dims > 3:
        ctx = pytest.raises(KeyError)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        """GhostBatchNorm{1,2,3}d should work, but other ints should throw."""
        module = ModuleWithBatchnorm(num_dims)
        assert module_surgery.count_module_instances(module, _GHOSTBN_MODULE_CLASS) == 0
        ghostbn.apply_ghost_batchnorm(module, ghost_batch_size=1)
        assert module_surgery.count_module_instances(module, _GHOSTBN_MODULE_CLASS) == 1


@pytest.mark.parametrize('num_dims', _TEST_NUM_DIMS)
@pytest.mark.parametrize('ghost_batch_size', _TEST_GHOST_BATCH_SIZES)
@pytest.mark.parametrize('batch_size', _TEST_BATCH_SIZES)
class TestGhostBatchesNormalized:

    def _assert_ghost_batches_normalized(self, module: ModuleWithBatchnorm, ghost_batch_size: int,
                                         batch_size: int) -> None:
        torch.manual_seed(123)
        size = [batch_size, module.num_features] + ([3] * module.num_dims)
        X = torch.randn(size=size)
        module.train()
        out = module(X)
        n_channels = out.shape[1]
        # reduce over everything but channel idx
        reduce_dims = (0,) + tuple(range(2, out.ndim))

        nchunks = int(math.ceil(batch_size / ghost_batch_size))
        for ghost_batch in out.chunk(nchunks):
            channel_variances, channel_means = torch.var_mean(ghost_batch, dim=reduce_dims, unbiased=False)
            torch.testing.assert_close(channel_variances, torch.ones(n_channels))
            torch.testing.assert_close(channel_means, torch.zeros(n_channels))

    def test_normalization_correct_functional(self, num_dims: int, ghost_batch_size: int, batch_size: int) -> None:
        module = ModuleWithBatchnorm(num_dims=num_dims)
        ghostbn.apply_ghost_batchnorm(module, ghost_batch_size=ghost_batch_size)
        self._assert_ghost_batches_normalized(module=module, ghost_batch_size=ghost_batch_size, batch_size=batch_size)

    def test_normalization_correct_algorithm(self, state, algo_instance, num_dims: int, ghost_batch_size: int,
                                             batch_size: int) -> None:
        algo_instance.apply(_GHOSTBN_CORRECT_EVENT, state, logger=Mock())
        module = cast(ModuleWithBatchnorm, state.model)
        self._assert_ghost_batches_normalized(module=module, ghost_batch_size=ghost_batch_size, batch_size=batch_size)


@pytest.mark.parametrize('ghost_batch_size', [4])
def test_correct_event_matches(algo_instance):
    assert algo_instance.match(_GHOSTBN_CORRECT_EVENT, Mock(side_effect=ValueError))


@pytest.mark.parametrize('ghost_batch_size', [4])
@pytest.mark.parametrize('event', Event)  # enum iteration
def test_incorrect_event_does_not_match(event: Event, algo_instance):
    if event == _GHOSTBN_CORRECT_EVENT:
        return
    assert not algo_instance.match(event, Mock(side_effect=ValueError))


@pytest.mark.parametrize('ghost_batch_size', [4])
@pytest.mark.parametrize('num_dims', [2])
def test_algorithm_logging(state, algo_instance):
    logger_mock = Mock()
    algo_instance.apply(Event.INIT, state, logger_mock)
    logger_mock.log_hyperparameters.assert_called_once_with({
        'GhostBatchNorm/num_new_modules': 1,
    })
