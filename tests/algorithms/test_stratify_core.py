# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pytest
import torch

from composer.algorithms.stratify_batches import stratify_core as core

# from composer.algorithms.stratify_batches import

@pytest.fixture
def simple_elems():
    return np.array([3, 1, 2, 3, 3, 2])  # one 1, two 2s, three 3s, zero 0s


@pytest.fixture
def uniq_elems():
    return np.array([3, 1, 2, 5, 8, 4]) # plausible indices


@pytest.fixture
def balanced_sampler(simple_elems):
    return core.BalancedSampler(simple_elems)


# what properties should balanced_sampler have?
#   -reset works
#   -counts of each sample always all within 1
#       -just within one epoch if replace=False
#       -across many epochs if replace=True
#   -throws if replace=False and try to read past epoch
#   -doesn't throw if replace=False and try to read exactly end of epoch
#   -doesn't throw ever if replace=True

def _equal_when_sorted(x: Iterable, y: Iterable):
    return np.array_equal(np.sort(x), np.sort(y))


def test_balanced_sampler_init(simple_elems):
    sampler = core.BalancedSampler(simple_elems)
    # resets/has correct data at init
    assert _equal_when_sorted(simple_elems, sampler.data)
    assert _equal_when_sorted(simple_elems, sampler.tail)


@pytest.mark.parametrize('num_vals', [0, 1, 2])
def test_balanced_sampler_sample(simple_elems, num_vals):
    sampler = core.BalancedSampler(simple_elems)
    # tail changes after sampling
    vals = sampler.sample(num_vals)
    if num_vals > 0: # if data should have actually changed
        assert not _equal_when_sorted(simple_elems, sampler.tail)
    # removed vals are ones it returned
    assert _equal_when_sorted(simple_elems, vals + list(sampler.tail))


def test_balanced_sampler_reset(simple_elems):
    sampler = core.BalancedSampler(simple_elems)
    assert sampler.is_reset()
    sampler.sample(len(simple_elems) - 1)
    assert len(sampler.tail) == 1
    assert not sampler.is_reset()
    sampler.reset()
    assert sampler.is_reset()
    assert _equal_when_sorted(simple_elems, sampler.tail)


def test_balanced_sampler_data_doesnt_change(simple_elems):
    original_simple_elems = simple_elems.copy()
    sampler = core.BalancedSampler(simple_elems)

    sampler.sample(1)
    assert _equal_when_sorted(original_simple_elems, sampler.data)
    # read all the data
    sampler.sample(len(original_simple_elems) - 1)
    assert _equal_when_sorted(original_simple_elems, sampler.data)
    # new epoch
    sampler.replace = True
    sampler.sample(1)
    assert _equal_when_sorted(original_simple_elems, sampler.data)


# def test_balanced_sampler_throws_past_end_if_no_replace(simple_elems):
#     sampler = core.BalancedSampler(simple_elems)
#     sampler.replace = False
#     sampler.sample(len(simple_elems))  # exactly end of epoch
#     with pytest.raises(IndexError):
#         sampler.sample(1)  # this is now past end of epoch

#     sampler.reset()
#     sampler.sample(len(simple_elems) + 1)  # immediate read past end

# @pytest.mark.parametrize('batch_size', [1])
# @pytest.mark.parametrize('num_batches', [8])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 6, 7, 8, 12, 13])
@pytest.mark.parametrize('num_batches', [1, 2, 3, 4])
def test_balanced_sampler_balances_sample_counts(uniq_elems, batch_size, num_batches):
    sampler = core.BalancedSampler(uniq_elems)
    # setup so we can easily increment counts and compare largest and smallest counts
    largest_elem = uniq_elems.max()
    counts = np.zeros(largest_elem + 1)
    elem_present_mask = np.zeros(largest_elem + 1, dtype=np.bool)
    elem_present_mask[uniq_elems] = True

    for _ in range(num_batches):
        idxs = np.asarray(sampler.sample(batch_size))
        for idx in idxs:
            counts[idx] += 1  # has to be serial to handle duplicates in idxs
        valid_counts = counts[elem_present_mask]
        assert valid_counts.max() - valid_counts.min() <= 1


@pytest.mark.parametrize('num_classes', [2, 4, 7, 8])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 7, 8])
@pytest.mark.parametrize('num_batches', [1, 2, 3])
def test_sample_batches_balanced_correctness(num_classes, batch_size, num_batches):
    pass  # TODO


@pytest.mark.parametrize('num_classes', [2, 4, 7, 8])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 7, 8])
@pytest.mark.parametrize('num_batches', [1, 2, 3])
def test_sample_batches_stratified_correctness(num_classes, batch_size, num_batches):
    pass
