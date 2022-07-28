# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, List, Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from composer.core import State
from composer.trainer.ddp import ddp_sync_context, prepare_ddp_module
from composer.utils import dist


class MinimalConditionalModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.choice1 = nn.Linear(1, 1, bias=False)
        self.choice2 = nn.Linear(1, 1, bias=False)
        self.choice3 = nn.Linear(1, 1, bias=False)

        nn.init.constant_(self.choice1.weight, 0)
        nn.init.constant_(self.choice2.weight, 0)
        nn.init.constant_(self.choice3.weight, 0)

    def forward(self, input: int):
        if input == 1:
            return self.choice1(Tensor([1]))
        if input == 2:
            return self.choice2(Tensor([1]))
        if input == 3:
            return self.choice3(Tensor([1]))
        raise Exception('Invalid input:', input)

    def loss(self, output: Tensor, target: Tensor):
        return (output - target) * (output - target)


@pytest.mark.parametrize('ddp_sync_strategy,expected_grads', [
    pytest.param('single_auto_sync', ([-1, None, None], [-1, -1.5, None], [-1, -1.5, None]), id='single_auto_sync'),
    pytest.param('multi_auto_sync', ([-1.5, None, None], [-1.5, -1.5, None], [-1.5, -1.5, None]), id='multi_auto_sync'),
    pytest.param('forced_sync', ([-1, None, None], [-1, -1, None], [-1.5, -1.5, None]), id='forced_sync'),
])
@pytest.mark.world_size(2)
def test_ddp_sync_strategy(ddp_sync_strategy: str, expected_grads: List[List[Optional[float]]],
                           dummy_train_dataloader: Iterable, rank_zero_seed: int):
    original_model = MinimalConditionalModel()
    # ddp = DDP(backend="gloo", find_unused_parameters=True, sync_strategy=ddp_sync_strategy, timeout=5.)
    optimizer = torch.optim.SGD(original_model.parameters(), 0.1)
    state = State(
        model=original_model,
        rank_zero_seed=rank_zero_seed,
        run_name='run_name',
        optimizers=optimizer,
        grad_accum=2,
        max_duration='1ep',
        dataloader=dummy_train_dataloader,
        dataloader_label='train',
        precision='fp32',
    )

    batches = [[(1, Tensor([1])), (1, Tensor([2]))], [(2, Tensor([1])), (2, Tensor([2]))]]
    state.model = prepare_ddp_module(state.model, find_unused_parameters=True)
    optimizer.zero_grad()

    for microbatch_idx in range(2):
        with ddp_sync_context(state, microbatch_idx == 1, sync_strategy=ddp_sync_strategy):
            input, target = batches[microbatch_idx][dist.get_local_rank()]

            output = state.model.forward(input)
            loss = original_model.loss(output, target)
            loss.mul_(1 / 2)
            loss.backward()

            if dist.get_global_rank() == 0:
                grads = [p.grad.item() if p.grad else None for p in original_model.parameters()]
                for expected, actual in zip(expected_grads[microbatch_idx], grads):
                    assert expected == actual

    if dist.get_global_rank() == 0:
        grads = [p.grad.item() if p.grad else None for p in original_model.parameters()]
        for expected, actual in zip(expected_grads[-1], grads):
            assert expected == actual
