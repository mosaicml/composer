# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.distributed._tensor import DTensor

from composer.utils.parallelism import FSDP2Config
from tests.common import (
    world_size,
)
from tests.trainer.fsdp2_context import fsdp2_context, prepare_fully_shard


class SimpleModel(nn.Module):
    """Simple model for testing FSDP2 with GradScaler.

    we need a submodule to test out FSDP2 wrapper.
    """

    def __init__(self, in_features=2, out_features=2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        return x


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
def test_fsdp2_with_gradscaler_inf(world_size: int):
    """Test FSDP2 with GradScaler can handle inf in grad."""
    # Setup GradScaler for mixed precision training
    scaler = GradScaler(enabled=True)

    # Choose dtype
    dtype = torch.float16

    model = SimpleModel().to('cuda')
    # Apply fully_shard to the model
    prepare_fully_shard(model, FSDP2Config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # dummy inputs and targets
    inputs = torch.randn(1, 2, device='cuda', dtype=dtype)
    targets = torch.randn(1, 2, device='cuda', dtype=dtype)

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass with autocast if using mixed precision
    with torch.amp.autocast(dtype=dtype, device_type='cuda'):
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)

    # Backward and optimize with gradient scaling
    scaler.scale(loss).backward()

    # manually hack rank 0 to have infs
    assert isinstance(model.linear1.weight.grad, DTensor)
    model.linear1.weight.grad.to_local()[:] = torch.ones_like(model.linear1.weight.grad.to_local())
    if dist.get_rank() == 0:
        model.linear1.weight.grad.to_local()[0, 0] = float('inf')
        assert torch.any(torch.isinf(model.linear1.weight.grad.to_local()))
    else:
        assert not torch.any(torch.isinf(model.linear1.weight.grad.to_local()))

    # Unscale gradients and check for infs/NaNs
    scaler.unscale_(optimizer)
    assert scaler._found_inf_per_device(optimizer), 'Found infs in gradients'

    # Step optimizer, since there is an inf, none of the ranks should update
    assert isinstance(model.linear1.weight, DTensor)
    prev_weight = model.linear1.weight.to_local().clone()
    scaler.step(optimizer)
    new_weight = model.linear1.weight
    assert torch.equal(prev_weight, new_weight.to_local())

    scaler.update()
