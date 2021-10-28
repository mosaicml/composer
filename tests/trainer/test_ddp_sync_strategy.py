# Copyright 2021 MosaicML. All Rights Reserved.

import os
from typing import List, Optional

import pytest
import torch
import torch.nn as nn

from composer.core.state import State
from composer.core.types import Tensor
from composer.trainer.ddp import DDP, FileStoreHparams
from composer.trainer.devices.device_cpu import DeviceCPU


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
        raise Exception("Invalid input:", input)

    def loss(self, output: Tensor, target: Tensor):
        return (output - target) * (output - target)


@pytest.mark.run_long
@pytest.mark.timeout(90)
@pytest.mark.parametrize("ddp_sync_strategy,expected_grads", [
    pytest.param('single_auto_sync', ([-1, None, None], [-1, -1.5, None], [-1, -1.5, None]), id='single_auto_sync'),
    pytest.param('multi_auto_sync', ([-1.5, None, None], [-1.5, -1.5, None], [-1.5, -1.5, None]), id='multi_auto_sync'),
    pytest.param('forced_sync', ([-1, None, None], [-1, -1, None], [-1.5, -1.5, None]), id='forced_sync'),
])
def test_ddp_sync_strategy(ddp_sync_strategy: str, expected_grads: List[Optional[float]], ddp_tmpdir: str):

    original_model = MinimalConditionalModel()

    device = DeviceCPU(num_cpus=2)

    ddp = DDP(nproc_per_node=device.nproc_per_node,
              store_hparams=FileStoreHparams(os.path.join(ddp_tmpdir, "store")),
              node_rank=0,
              num_nodes=1,
              backend=device.ddp_backend,
              fork_rank_0=True,
              find_unused_parameters=True,
              ddp_sync_strategy=ddp_sync_strategy)

    optimizer = torch.optim.SGD(original_model.parameters(), 0.1)

    state = State(model=original_model,
                  optimizers=optimizer,
                  train_batch_size=1,
                  eval_batch_size=1,
                  grad_accum=2,
                  max_epochs=1,
                  precision='fp32',
                  world_size=2,
                  nproc_per_node=2)

    batches = [[(1, Tensor([1])), (1, Tensor([2]))], [(2, Tensor([1])), (2, Tensor([2]))]]

    def basic_train_loop():
        state.model = ddp.prepare_module(state.model)

        optimizer.zero_grad()

        for microbatch_idx in range(2):
            with ddp.ddp_sync_context(state, microbatch_idx == 1):
                input, target = batches[microbatch_idx][state.local_rank]

                output = state.model.forward(input)
                loss = original_model.loss(output, target)
                loss.mul_(1 / 2)
                loss.backward()

                if state.is_rank_zero:
                    grads = [p.grad.item() if p.grad else None for p in original_model.parameters()]
                    for expected, actual in zip(expected_grads[microbatch_idx], grads):
                        assert expected == actual

        if state.is_rank_zero:
            grads = [p.grad.item() if p.grad else None for p in original_model.parameters()]
            for expected, actual in zip(expected_grads[-1], grads):
                assert expected == actual

    ddp.launch(state, basic_train_loop)
