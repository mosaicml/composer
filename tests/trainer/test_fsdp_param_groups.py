# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import gc

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist, misc
from tests.common import RandomClassificationDataset, SimpleModel, device, world_size


@pytest.mark.parametrize('mixed_precision', ['DEFAULT'])
@pytest.mark.parametrize('reentrant', [True, False])
@pytest.mark.filterwarnings('ignore::UserWarning')
@device('gpu')
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2'),
                    reason='FSDP use_orig_params requires torch 2.0 or higher')
def test_fsdp_param_groups_without_orig_params(mixed_precision: str, device: str, reentrant: bool, world_size: int):
    """

    Ensure that FSDP with 'use_orig_params=False' raises an exception when passing in an optimizer
    with multiple param groups

    """
    num_classes = 10
    model = SimpleModel(num_features=1, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))

    # create a different parameter per group
    param_groups = [{'params': param, 'lr': (0.1 + 0.1 * i)} for i, param in enumerate(model.parameters())]
    optimizer = torch.optim.SGD(param_groups, lr=0)

    expected_error = 'Multiple optimizer groups with FSDP are only supported on torch 2.0 \
                                   with use_orig_params=True.'

    with pytest.raises(RuntimeError, match=expected_error):
        _ = Trainer(model=model,
                    optimizers=optimizer,
                    train_dataloader=dataloader,
                    fsdp_config={
                        'activation_checkpointing_reentrant': reentrant,
                        'mixed_precision': mixed_precision,
                        'use_orig_params': False
                    },
                    max_duration='3ba',
                    device=device)
    gc.collect()


@pytest.mark.parametrize('mixed_precision', ['FULL', 'DEFAULT', 'PURE'])
@pytest.mark.parametrize('reentrant', [True, False])
@pytest.mark.filterwarnings('ignore::UserWarning')
@device('gpu')
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2'),
                    reason='FSDP use_orig_params requires torch 2.0 or higher')
def test_fsdp_with_param_groups(mixed_precision: str, device: str, reentrant: bool, world_size: int):
    """
    Test whether an optimizer with multiple param groups maintains the same param groups when
    wrapped with FSDP.
    We assert that the model is FSDP, and that the shapes and LRs of each
    FSDP wrapped-parameter match the unwrapped parameter, while pointing to different underlying weight
    tensors
    """
    num_classes = 10
    model = SimpleModel(num_features=1, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))

    # create a different group per parameter
    param_groups = [{'params': param, 'lr': (0.1 + 0.1 * i)} for i, param in enumerate(model.parameters())]
    optimizer = torch.optim.SGD(param_groups, lr=0)

    unwrapped_optimizer = copy.deepcopy(optimizer)

    optimizer_groups_pre_fsdp = optimizer.param_groups

    trainer = Trainer(model=model,
                      optimizers=optimizer,
                      train_dataloader=dataloader,
                      fsdp_config={
                          'activation_checkpointing_reentrant': reentrant,
                          'mixed_precision': mixed_precision
                      },
                      max_duration='3ba',
                      device=device)
    trainer.fit()

    assert misc.is_model_fsdp(trainer.state.model)
    trainer_optimizer = trainer.state.optimizers[0]
    assert len(trainer_optimizer.param_groups) > 1
    assert len(trainer_optimizer.param_groups) == len(optimizer_groups_pre_fsdp)

    with trainer.state.model.module.summon_full_params(trainer.state.model.module):  # type: ignore
        for unwrapped_param_group, wrapped_param_group in zip(unwrapped_optimizer.param_groups,
                                                              trainer_optimizer.param_groups):

            unwrapped_param_list = unwrapped_param_group['params']
            wrapped_param_list = wrapped_param_group['params']

            assert len(unwrapped_param_list) == 1
            assert len(wrapped_param_list) == 1

            unwrapped_param = unwrapped_param_list[0]
            wrapped_param = wrapped_param_list[0]

            assert unwrapped_param.shape == wrapped_param.shape

            # the underlying tensor is different because it has been recreated when FSDP wraps the model
            assert id(unwrapped_param) != id(wrapped_param)

            assert unwrapped_param_group['lr'] == wrapped_param_group['lr']
