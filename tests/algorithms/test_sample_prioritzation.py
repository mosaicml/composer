# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
import torch
from maskedtensor import masked_tensor

from composer.algorithms.sample_prioritization.sample_prioritization import (SamplePrioritization, keep_logic,
                                                                             masked_l2, masked_max, masked_mean,
                                                                             masked_median, masked_min,
                                                                             select_using_loss)
from composer.core import Algorithm, Event
from composer.core.state import State
from composer.loggers import Logger
from composer.models import ComposerClassifier


class TestKeepLogic:
    test_tensor = torch.tensor([0, 1, 2, 3])
    n_select = 2

    @pytest.mark.parametrize('keep_from', ['bottom', 'top', 'middle'])
    def test_keep_logic(self, keep_from, sorted_idx=test_tensor, n_select=n_select):
        select_idx = keep_logic(keep_from, sorted_idx, n_select)
        if keep_from == 'bottom':
            assert torch.all(select_idx == torch.Tensor([True, True, False, False])).item()
        elif keep_from == 'top':
            assert torch.all(select_idx == torch.Tensor([False, False, True, True])).item()
        else:
            assert torch.all(select_idx == torch.Tensor([True, False, False, True])).item()


class TestMaskedMetrics:

    test_tensor = torch.tensor([[0., 1., 1., 0.], [1., -1., 0., 0.], [0., 1., 1., 1.]])

    @pytest.mark.parametrize(
        'mskd_tensor', [masked_tensor(test_tensor, test_tensor > 0),
                        masked_tensor(test_tensor, test_tensor >= 0)])
    def test_masked_median(self, mskd_tensor):
        output = masked_median(mskd_tensor)
        if mskd_tensor.mask()[0, 0]:
            assert torch.all(output == torch.Tensor([0, 0, 1])).item()
        else:
            assert torch.all(output == torch.Tensor([1, 1, 1])).item()

    @pytest.mark.parametrize(
        'mskd_tensor', [masked_tensor(test_tensor, test_tensor > 0),
                        masked_tensor(test_tensor, test_tensor >= 0)])
    def test_masked_mean(self, mskd_tensor):
        output = masked_mean(mskd_tensor)
        if mskd_tensor.mask()[0, 0]:
            assert torch.all(output == torch.Tensor([0.5, (1 / 3), 0.75])).item()
        else:
            assert torch.all(output == torch.Tensor([1, 1, 1])).item()

    @pytest.mark.parametrize(
        'mskd_tensor', [masked_tensor(test_tensor, test_tensor > 0),
                        masked_tensor(test_tensor, test_tensor >= 0)])
    def test_masked_l2(self, mskd_tensor):
        output = masked_l2(mskd_tensor)
        assert torch.all(output == torch.sqrt(torch.Tensor([2., 1., 3.]))).item()

    @pytest.mark.parametrize(
        'mskd_tensor', [masked_tensor(test_tensor, test_tensor > 0),
                        masked_tensor(test_tensor, test_tensor >= 0)])
    def test_masked_max(self, mskd_tensor):
        output = masked_max(mskd_tensor)
        assert torch.all(output == torch.Tensor([1, 1, 1])).item()

    @pytest.mark.parametrize(
        'mskd_tensor', [masked_tensor(test_tensor, test_tensor > 0),
                        masked_tensor(test_tensor, test_tensor >= 0)])
    def test_masked_min(self, mskd_tensor):
        output = masked_min(mskd_tensor)
        if mskd_tensor.mask()[0, 0]:
            assert torch.all(output == torch.Tensor([0, 0, 0])).item()
        else:
            assert torch.all(output == torch.Tensor([1, 1, 1])).item()


@pytest.mark.filterwarnings(
    r'ignore:Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer:UserWarning:torchmetrics')
class TestSelectUsingLoss:

    def test_select_using_loss(self, synthetic_hf_state):
        state = synthetic_hf_state[0]
        sample = next(iter(state.dataloader)).data
        sample_sizes = {k: v.size() for k, v in sample.items()}
        labels = sample.pop('labels')
        pct_keep = 0.5
        new_input, new_target = select_using_loss(sample,
                                                  target=labels,
                                                  model=state.model,
                                                  pct_keep=pct_keep,
                                                  selection_metric='max',
                                                  keep_from='bottom')
        for k, v in sample_sizes.items():
            if k == 'labels':
                assert new_target.size() == torch.Size([int(pct_keep * v[0]), v[1]])
            else:
                assert new_input[k].size() == torch.Size([int(pct_keep * v[0]), v[1]])


class TestSamplePrioritization:

    def test_sample_prioritization(self, synthetic_hf_state, empty_logger):
        state = synthetic_hf_state[0]
        state.batch = next(iter(state.dataloader)).data
        sample_sizes = {k: v.size() for k, v in state.batch.items()}
        pct_keep = 0.5
        algorithm = SamplePrioritization(pct_keep=pct_keep)
        algorithm.apply(Event.BEFORE_FORWARD, state, empty_logger)
        for k, v in sample_sizes.items():
            assert state.batch[k].size() == torch.Size([int(pct_keep * v[0]), v[1]])
