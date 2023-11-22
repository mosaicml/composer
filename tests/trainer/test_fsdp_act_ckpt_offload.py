# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from composer import Trainer
from composer.models import ComposerModel
from tests.common import world_size


class SimpleModel(ComposerModel):

    def __init__(self, num_features: int = 128, device: str = 'cuda'):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc2 = torch.nn.Linear(num_features, num_features, device=device, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.ReLU(x)
        x = self.fc2(x)
        return x

    def loss(self, outputs, batch):
        pass


@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('activation_checkpointing', [True, False])
@pytest.mark.parametrize('activation_cpu_offload', [True, False])
def test_fsdp_act_ckpt_offload(
    activation_checkpointing: bool,
    activation_cpu_offload: bool,
    world_size: int,
):
    model = (SimpleModel())

    fsdp_config = {
        'activation_checkpointing': activation_checkpointing,
        'activation_checkpointing_reentrant': False,
        'activation_cpu_offload': activation_cpu_offload,
    }

    model.fc1._activation_checkpointing = True

    trainer = Trainer(
        model=model,
        device='gpu',
        fsdp_config=fsdp_config,
    )

    assert trainer.state.fsdp_enabled
    if version.parse(torch.__version__) > version.parse('2.1.0.dev'):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import OffloadWrapper

        if activation_checkpointing and activation_cpu_offload:
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module, OffloadWrapper)
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module._checkpoint_wrapped_module,
                              CheckpointWrapper)
        elif activation_checkpointing:
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module, CheckpointWrapper)
        elif activation_cpu_offload:
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module, OffloadWrapper)
        else:
            assert not isinstance(trainer.state.model.fc1._fsdp_wrapped_module, CheckpointWrapper)
