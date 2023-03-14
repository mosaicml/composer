# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.nn import GroupNorm
from torch.utils.data import DataLoader

from composer.algorithms.low_precision_groupnorm import LowPrecisionGroupNorm, apply_low_precision_groupnorm
from composer.algorithms.low_precision_groupnorm.low_precision_groupnorm import LPGroupNorm
from composer.core import Event, State
from composer.loggers import Logger
from composer.utils import get_device
from tests.common import RandomImageDataset
from tests.common.models import SimpleConvModel


def assert_is_lpgn_instance(model):
    # ensure that within the entire model, no PyTorch GroupNorm exists, and at least one LPGN does.
    assert model.modules is not None, 'model has .modules method'
    for module_class in model.modules():
        if isinstance(module_class, GroupNorm):
            assert isinstance(module_class, LPGroupNorm)

    assert any(isinstance(module_class, LPGroupNorm) for module_class in model.modules())


@pytest.mark.parametrize('affine', [True, False])
def test_low_precision_groupnorm_functional(affine):
    model = SimpleConvModel(norm='group', norm_affine=affine)
    dataloader = DataLoader(RandomImageDataset(), batch_size=2)
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
        precision='amp_fp16',
        device=get_device('cpu'),
    )

    apply_low_precision_groupnorm(state.model, state._precision, state.optimizers)
    assert_is_lpgn_instance(state.model)


@pytest.mark.parametrize('affine', [True, False])
def test_low_precision_groupnorm_algorithm(affine, empty_logger: Logger):
    model = SimpleConvModel(norm='group', norm_affine=affine)
    dataloader = DataLoader(RandomImageDataset(), batch_size=2)

    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
        precision='amp_fp16',
        device=get_device('cpu'),
    )
    low_precision_groupnorm = LowPrecisionGroupNorm()

    low_precision_groupnorm.apply(Event.INIT, state, empty_logger)

    assert_is_lpgn_instance(state.model)
