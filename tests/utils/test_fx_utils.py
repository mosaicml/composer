# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import operator

import pytest
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule
from torchvision import models

from composer.utils.fx_utils import apply_stochastic_residual, count_op_instances, fuse_parallel_linears, replace_op


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
        (MyTestModel, nn.ReLU, 1),
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


class SimpleParallelLinears(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        y = self.fc1(x)
        z = self.fc2(x)
        return y + z


class ParallelLinears(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.ln = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        y = self.fc1(x)
        y = self.ln(y)
        y = self.relu(y)
        z = self.fc2(x)
        return y + z


class NotFusibleLinears(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=False)
        self.ln = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        y = self.fc1(x)
        y = self.ln(y)
        y = self.relu(y)
        z = self.fc2(x)
        return y + z


class NotParallelLinears(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.ln = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        y = self.fc1(x)
        y = self.ln(y)
        y = self.relu(y)
        z = self.fc2(y)
        return x + z


# Incorrect warning fixed in https://github.com/pytorch/pytorch/pull/61463
@pytest.mark.parametrize(
    'model_cls, before_count, after_count',
    [
        (SimpleParallelLinears, 2, 1),
        (ParallelLinears, 2, 1),
        (NotParallelLinears, 2, 2),
        (NotFusibleLinears, 2, 2),
    ],
)
@pytest.mark.filterwarnings(
    r'ignore:Attempted to insert a call_module Node with no underlying reference in the owning GraphModule!.*:UserWarning'
)
def test_fuse_parallel_linears(model_cls, before_count, after_count):
    model = model_cls()
    traced = symbolic_trace(model)

    assert isinstance(traced, GraphModule)

    assert count_op_instances(traced, nn.Linear) == before_count

    fuse_parallel_linears(traced)

    assert count_op_instances(traced, nn.Linear) == after_count


@pytest.mark.parametrize(
    'model_cls, block_count',
    [(models.resnet18, 8)],
)
@pytest.mark.filterwarnings(
    r'ignore:Attempted to insert a call_module Node with no underlying reference in the owning GraphModule!.*:UserWarning'
)
def test_stochastic_depth(model_cls, block_count):
    model = model_cls()
    traced = symbolic_trace(model)

    assert isinstance(traced, GraphModule)

    inp = torch.randn(1, 3, 224, 224)

    traced_st_depth_no_drop, residual_count = apply_stochastic_residual(traced, 0.0)

    out_traced = traced(inp)
    out_traced_st_depth_no_drop = traced_st_depth_no_drop(inp)
    assert torch.allclose(out_traced,
                          out_traced_st_depth_no_drop), 'mismatch in outputs with 0 drop rate for stochastic modules'
    assert residual_count == block_count
