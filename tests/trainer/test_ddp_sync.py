import os

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
def test_ddp_sync_strategy(ddp_tmpdir: str):

    import logging

    logging.getLogger().setLevel(0)

    print('entering test')

    if "RANK" in os.environ:
        # raise Exception("foo")
        print('with rank', os.environ["RANK"])
        pass

    original_model = MinimalConditionalModel()

    device = DeviceCPU(num_cpus=2)

    ddp = DDP(
        nproc_per_node=device.nproc_per_node,
        store_hparams=FileStoreHparams(os.path.join(ddp_tmpdir, "store")),
        node_rank=0,
        num_nodes=1,
        backend=device.ddp_backend,
        fork_rank_0=True,
        find_unused_parameters=True,
        # ddp_sync_strategy='single_auto_sync')
        ddp_sync_strategy='multi_auto_sync')
    # ddp_sync_strategy='manual_sync')

    state = State(model=original_model,
                  optimizers=torch.optim.SGD(original_model.parameters(), 0.1),
                  train_batch_size=1,
                  eval_batch_size=1,
                  grad_accum=2,
                  max_epochs=1,
                  precision='fp32',
                  world_size=2,
                  nproc_per_node=2)

    batches = [[(1, Tensor([1])), (1, Tensor([2]))], [(2, Tensor([2])), (2, Tensor([1]))]]

    def basic_train_loop():
        state.model = ddp.prepare_module(state.model)

        state.optimizers.zero_grad()

        for microbatch_idx in range(2):
            ddp_sync_context = ddp.get_ddp_sync_context(state, microbatch_idx == 1)
            with ddp_sync_context():
                input, target = batches[microbatch_idx][state.local_rank]

                output = state.model.forward(input)
                loss = original_model.loss(output, target)
                loss.mul_(1 / 2)
                loss.backward()

                print([p.grad for p in original_model.parameters()])

        print([p.grad for p in original_model.parameters()])
        print(state.optimizers.param_groups[0]['params'])
        raise Exception

    ddp.launch(state, basic_train_loop)
