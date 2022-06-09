# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import torch

from composer.algorithms import factorize


@dataclass
class _RankReduce(object):
    """This is just here for convenience when testing."""
    batch_size: int = 100
    C_out: int = 64
    C_in: int = 32
    C_latent_now: int = 16
    C_latent_new: int = 8
    n_iters: int = 2
    seed: int = 123
    op: str = 'linear'

    def __call__(self, already_factorized=False):
        torch.manual_seed(self.seed)
        X = torch.randn(self.batch_size, self.C_in)
        bias = torch.randn(self.C_out)
        if already_factorized:
            Wa = torch.randn(self.C_in, self.C_latent_now)
            Wb = torch.randn(self.C_latent_now, self.C_out)
            Y = (X @ Wa) @ Wb + bias
        else:
            Wa = torch.randn(self.C_in, self.C_out)
            Wb = None
            Y = X @ Wa + bias

        return factorize.factorize_matrix(X, Y, Wa, Wb, bias=bias, rank=self.C_latent_new, n_iters=self.n_iters)


@dataclass
class _RankReduceConv2d(object):
    """This is just here for convenience when testing."""
    batch_size: int = 1
    H: int = 4
    W: int = 4
    kernel_size: Tuple[int, int] = (3, 3)
    C_in: int = 32
    C_latent_now: int = 16
    C_latent_new: int = 6
    C_out: int = 24
    n_iters: int = 2
    seed: int = 123
    op: str = 'conv2d'

    def __call__(self, already_factorized=False):
        torch.manual_seed(self.seed)
        X = torch.randn(1, self.C_in, self.H, self.W)  # NCHW
        if already_factorized:
            Wa = torch.randn(self.C_latent_now, self.C_in, *self.kernel_size)
            Wb = torch.randn(self.C_out, self.C_latent_now, 1, 1)
            biasA = torch.randn(self.C_latent_now)
            biasB = torch.randn(self.C_out)
        else:
            Wa = torch.randn(self.C_out, self.C_in, *self.kernel_size)
            Wb = None
            biasA = torch.randn(self.C_out)
            biasB = None

        return factorize.factorize_conv2d(X,
                                          Wa,
                                          Wb,
                                          biasA=biasA,
                                          biasB=biasB,
                                          rank=self.C_latent_new,
                                          n_iters=self.n_iters)


@pytest.fixture(params=[_RankReduce(), _RankReduceConv2d()])
def factorize_task(request):
    return request.param


def _check_factorization(f: Union[_RankReduce, _RankReduceConv2d],
                         prev_nmse: Optional[float] = None,
                         already_factorized: bool = True):
    info = f(already_factorized=already_factorized)
    Wa = info.Wa
    Wb = info.Wb
    bias = info.bias  # one bias because only 2nd op needs one
    nmse = info.nmse

    op = f.op
    if op == 'linear':
        in_dim = 0
        out_dim = 1
    elif op == 'conv2d':
        in_dim = 1
        out_dim = 0
    else:
        raise ValueError('Invalid op: ', op)

    k = f.C_latent_new
    if k >= f.C_in or k >= f.C_out:
        # no point in factorizing with latent dim bigger than
        # either input or output; so just regress input onto output
        assert Wa is not None
        assert Wb is None
        assert Wa.shape[in_dim] == f.C_in
        assert Wa.shape[out_dim] == f.C_out
    elif k >= f.C_latent_now:
        # no need to factorize any futher than current factorization
        assert Wa is not None
        assert Wa.shape[in_dim] == f.C_in
        assert Wa.shape[out_dim] == f.C_latent_now
    else:
        # actually needed to factorize
        assert bias is not None
        assert Wa is not None
        assert Wb is not None
        assert bias.ndim == 1
        assert bias.shape[0] == f.C_out
        assert Wa.shape[in_dim] == f.C_in
        assert Wa.shape[out_dim] == f.C_latent_new
        assert Wb is not None  # should have actually factorized weights
        assert Wb.shape[in_dim] == f.C_latent_new
        assert Wb.shape[out_dim] == f.C_out

    assert nmse < 1.0  # should explain variance better than just predicting mean
    if prev_nmse is not None:
        assert nmse <= prev_nmse + 1e-8  # error decreases over time
    return nmse  # new "previous" nmse


@pytest.mark.parametrize(
    'shapes',
    [
        (16, 16, 16, 16),  # all the same
        (16, 8, 16, 16),  # already low rank
        (16, 8, 16, 16),  # requested rank > current latent rank
        (16, 16, 32, 16),  # requested rank > input rank
        (16, 16, 16, 8),  # requested rank > output rank
        (32, 16, 16, 16),  # requested rank >= output rank, and underdetermined
    ])
@pytest.mark.parametrize('already_factorized', [False, True])
def test_factorize_edge_cases(shapes, factorize_task, already_factorized):
    """Test edge cases regarding current and requested matrix shapes."""
    C_in, C_latent_now, C_latent_new, C_out = shapes
    factorize_task.C_in = C_in
    factorize_task.C_latent_now = C_latent_now
    factorize_task.C_latent_new = C_latent_new
    factorize_task.C_out = C_out
    _check_factorization(factorize_task, already_factorized=already_factorized)


@pytest.mark.parametrize('already_factorized', [False, True])
def test_factorize_more_dims_better(factorize_task, already_factorized):
    """More latent dimensions should yield nonincreasing error."""
    prev_nmse = np.inf
    for C_latent_new in [1, 4, 16, 32]:
        factorize_task.C_latent_new = C_latent_new
        maybe_nmse = _check_factorization(factorize_task, prev_nmse, already_factorized=already_factorized)
        prev_nmse = maybe_nmse if maybe_nmse else prev_nmse


@pytest.mark.parametrize('already_factorized', [False, True])
def test_factorize_more_iters_better(factorize_task, already_factorized):
    """More optimization iters should yield nonincreasing error."""
    prev_nmse = np.inf
    for n_iters in [0, 1, 2, 4]:
        factorize_task.n_iters = n_iters
        maybe_nmse = _check_factorization(factorize_task, prev_nmse, already_factorized=already_factorized)
        prev_nmse = maybe_nmse if maybe_nmse else prev_nmse
