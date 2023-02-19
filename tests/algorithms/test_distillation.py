# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest
import torch

from composer.algorithms.distillation import KLDivergence


def _matching_tensors() -> Tuple[torch.Tensor, torch.Tensor]:
    N = 64
    C = 10
    targets = torch.rand(N, C)
    return targets, torch.clone(targets)


def _non_matching_tensors() -> Tuple[torch.Tensor, torch.Tensor]:
    N = 64
    C = 10
    y_t = torch.rand(N, C)
    y_s = torch.rand(N, C)
    return y_t, y_s


@pytest.mark.parametrize('tensors', [_matching_tensors()])
def test_kl_same(tensors: Tuple[torch.Tensor, torch.Tensor]):
    y_t, y_s = tensors
    kl = KLDivergence()
    assert torch.isclose(kl(y_t, y_s), torch.tensor(0.0), atol=1e-5)


@pytest.mark.parametrize('tensors', [_non_matching_tensors()])
@pytest.mark.xfail
def test_kl_diff(tensors: Tuple[torch.Tensor, torch.Tensor]):
    y_t, y_s = tensors
    kl = KLDivergence()
    assert torch.isclose(kl(y_t, y_s), torch.tensor(0.0), atol=1e-5)


# class TestDistillationAlgorithm:

#     def test_
