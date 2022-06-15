# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import operator

import pytest
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule

from composer.utils.fx_utils import count_op_instances, replace_op


class MyTestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.factor = 0.5

    def forward(self, x):
        x = torch.add(x, self.factor)
        return self.relu(x)


class AddModel(nn.Module):

    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)


@pytest.mark.parametrize(
    'model_cls, ops, count',
    [
        (AddModel, operator.add, 1),
        (AddModel, [operator.add, torch.add], 2),
        (AddModel, [operator.add, torch.add, 'add'], 3),
    ],
)
def test_count_op_instances(model_cls, ops, count):
    model = model_cls()
    traced = symbolic_trace(model)

    assert isinstance(traced, GraphModule)

    assert count_op_instances(traced, ops) == count


@pytest.mark.parametrize(
    'model_cls, src_ops, tgt_op, count',
    [
        (MyTestModel, torch.add, torch.mul, 1),
    ],
)
def test_replace_op(model_cls, src_ops, tgt_op, count):
    model = model_cls()
    traced = symbolic_trace(model)

    assert isinstance(traced, GraphModule)

    replace_op(traced, src_ops, tgt_op)

    assert count_op_instances(traced, tgt_op) == count
