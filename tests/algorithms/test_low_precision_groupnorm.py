# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from torch.nn import LayerNorm
from torch.utils.data import DataLoader

from composer.algorithms.low_precision_groupnorm import LowPrecisionGroupNorm, apply_low_precision_groupnorm
from composer.algorithms.low_precision_groupnorm.low_precision_groupnorm import LPGroupNorm
from composer.core import Event, State
from composer.loggers import Logger
from composer.utils import get_device
from tests.common import RandomImageDataset
from tests.common.models import ConvModel


def assert_is_lpgn_instance(model):
    # ensure that within the entire model, no PyTorch GroupNorm exists, and at least one LPGN does.
    assert model.modules is not None, 'model has .modules method'
    for module_class in model.modules():
        if isinstance(module_class, LayerNorm):
            assert isinstance(module_class, LPGroupNorm)

    assert any(isinstance(module_class, LPGroupNorm) for module_class in model.modules())


def test_low_precision_groupnorm_functional(device: str):
    model = ConvModel(norm='groupnorm')
    dataloader = DataLoader(RandomImageDataset(), batch_size=2)
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='run_name',
        device=get_device(device),
        dataloader=dataloader,
        dataloader_label='train',
        max_duration='1ep',
        precision='amp_fp16',
    )

    apply_low_precision_groupnorm(state.model, state.optimizers, state._precision)
    assert_is_lpgn_instance(state.model)


def test_low_precision_groupnorm_algorithm(empty_logger: Logger, device: str):
    model = ConvModel(norm='groupnorm')
    dataloader = DataLoader(RandomImageDataset(), batch_size=2)

    state = State(model=model,
                  rank_zero_seed=0,
                  run_name='run_name',
                  device=get_device(device),
                  dataloader=dataloader,
                  dataloader_label='train',
                  max_duration='1ep',
                  precision='amp_fp16')
    low_precision_groupnorm = LowPrecisionGroupNorm()

    low_precision_groupnorm.apply(Event.INIT, state, empty_logger)

    assert_is_lpgn_instance(state.model)
