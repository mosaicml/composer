# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy

from composer.core import Evaluator, State
from composer.core.types import DataLoader
from composer.trainer.ddp import _ddp_sync_context, _prepare_ddp_module
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
        raise Exception("Invalid input:", input)

    def loss(self, output: Tensor, target: Tensor):
        return (output - target) * (output - target)


@pytest.mark.timeout(90)
@pytest.mark.parametrize("ddp_sync_strategy,expected_grads", [
    pytest.param('single_auto_sync', ([-1, None, None], [-1, -1.5, None], [-1, -1.5, None]), id='single_auto_sync'),
    pytest.param('multi_auto_sync', ([-1.5, None, None], [-1.5, -1.5, None], [-1.5, -1.5, None]), id='multi_auto_sync'),
    pytest.param('forced_sync', ([-1, None, None], [-1, -1, None], [-1.5, -1.5, None]), id='forced_sync'),
])
@pytest.mark.world_size(2)
def test_ddp_sync_strategy(ddp_sync_strategy: str, expected_grads: List[Optional[float]],
                           dummy_train_dataloader: DataLoader, dummy_val_dataloader: DataLoader):
    original_model = MinimalConditionalModel()
    # ddp = DDP(backend="gloo", find_unused_parameters=True, sync_strategy=ddp_sync_strategy, timeout=5.)
    optimizer = torch.optim.SGD(original_model.parameters(), 0.1)
    metric_coll = MetricCollection([Accuracy()])
    evaluators = [Evaluator(label="dummy_label", dataloader=dummy_val_dataloader, metrics=metric_coll)]
    state = State(model=original_model,
                  rank_zero_seed=0,
                  optimizers=optimizer,
                  grad_accum=2,
                  max_duration="1ep",
                  train_dataloader=dummy_train_dataloader,
                  evaluators=evaluators,
                  precision='fp32')

    batches = [[(1, Tensor([1])), (1, Tensor([2]))], [(2, Tensor([1])), (2, Tensor([2]))]]
    state.model = _prepare_ddp_module(state.model, find_unused_parameters=True)
    optimizer.zero_grad()

    for microbatch_idx in range(2):
        with _ddp_sync_context(state, microbatch_idx == 1, sync_strategy=ddp_sync_strategy):
            input, target = batches[microbatch_idx][dist.get_local_rank()]

            output = state.model.forward(input)
            loss = original_model.loss(output, target)
            loss.mul_(1 / 2)
            loss.backward()

            if dist.get_global_rank() == 0:
                grads = [p.grad.item() if p.grad else None for p in original_model.parameters()]
                for expected, actual in zip(expected_grads[microbatch_idx], grads):  # type: ignore
                    assert expected == actual

    if dist.get_global_rank() == 0:
        grads = [p.grad.item() if p.grad else None for p in original_model.parameters()]
        for expected, actual in zip(expected_grads[-1], grads):  # type: ignore
            assert expected == actual
