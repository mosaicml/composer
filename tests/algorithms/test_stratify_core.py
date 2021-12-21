# Copyright 2021 MosaicML. All Rights Reserved.

import math
from typing import Iterable, List, Sequence

import numpy as np
import pytest

from composer.algorithms.stratify_batches import stratify_core as core


@pytest.fixture
def simple_elems():
    return np.array([3, 1, 2, 3, 3, 2])  # one 1, two 2s, three 3s, zero 0s


@pytest.fixture
def uniq_elems():
    return np.array([3, 1, 2, 5, 8, 4])  # plausible indices


@pytest.fixture
def balanced_sampler(simple_elems):
    return core.BalancedSampler(simple_elems)


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
    if num_vals > 0:  # if data should have actually changed
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


def test_balanced_sampler_no_replace_throws_on_excess_count(simple_elems):
    sampler = core.BalancedSampler(simple_elems, replace=False)
    with pytest.raises(ValueError):
        sampler.sample(len(simple_elems) + 1)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 6, 7, 8, 12, 13])
@pytest.mark.parametrize('num_batches', [1, 2, 3, 4])
def test_balanced_sampler_balances_sample_counts(uniq_elems, batch_size, num_batches):
    sampler = core.BalancedSampler(uniq_elems, replace=True)
    # setup so we can easily increment counts and compare largest and smallest counts
    largest_elem = uniq_elems.max()
    counts = np.zeros(largest_elem + 1)
    elem_present_mask = np.zeros(largest_elem + 1, dtype=bool)
    elem_present_mask[uniq_elems] = True

    for _ in range(num_batches):
        idxs = np.asarray(sampler.sample(batch_size))
        for idx in idxs:
            counts[idx] += 1  # can't just do counts[idxs] because of duplicates
        valid_counts = counts[elem_present_mask]
        assert valid_counts.max() - valid_counts.min() <= 1


def _make_targets(num_classes: int, batch_size: int, num_batches: int, add_stragglers: bool, rng: np.random.Generator):
    num_targets = batch_size * num_batches
    if add_stragglers:
        num_targets += int(math.ceil(batch_size / 2))
    return rng.choice(num_classes, size=num_targets, replace=True)


def _construct_class_counts(batches: List[List], targets: Sequence[int], num_classes: int):
    targets = np.asarray(targets)  # needs to be ints starting at 0
    batched_idxs = np.array(batches)  # each row is indices for one batch
    batched_targets = targets[batched_idxs.ravel()].reshape(batched_idxs.shape)
    class_counts = np.zeros((len(batches), num_classes))
    for b, row in enumerate(batched_targets):
        class_counts[b] = np.bincount(row, minlength=num_classes)
    return class_counts


@pytest.mark.parametrize('num_classes', [2, 4, 7, 8])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 7, 8])
@pytest.mark.parametrize('num_batches', [1, 2, 3])
@pytest.mark.parametrize('add_stragglers', [False, True])
@pytest.mark.parametrize('stratify_how', ['balance', 'match'])
@pytest.mark.parametrize('seed', [123])
def test_sample_batches_correctness(num_classes: int, batch_size: int, num_batches: int, add_stragglers: bool,
                                    stratify_how: str, seed: int):
    rng = np.random.default_rng(seed)
    targets = _make_targets(num_classes=num_classes,
                            batch_size=batch_size,
                            num_batches=num_batches,
                            add_stragglers=add_stragglers,
                            rng=rng)
    dataset = np.zeros(len(targets))

    batches = iter(
        core.StratifiedBatchSampler(dataset,
                                    targets=targets,
                                    batch_size=batch_size,
                                    stratify_how=stratify_how,
                                    seed=seed))
    batches = np.array(list(batches))

    # check stragglers if applicable
    if add_stragglers:
        last_batch = batches[-1]
        assert len(last_batch) == len(batches[0])
        batches = batches[:-1]

    # sanity check batches
    assert len(list(batches)) == num_batches
    for b, batch in enumerate(batches):
        assert len(batch) == batch_size, f"batch {b}/{num_batches} has wrong size!"
        assert batch.min() >= 0, f"batch {b}/{num_batches} has invalid idxs!"
        assert batch.max() < len(targets), f"batch {b}/{num_batches} has invalid idxs!"

    # check balance properties
    class_counts = _construct_class_counts(batches, targets, num_classes=num_classes)
    # not all classes necessarily show up in the targets, so we need to not
    # throw because of these classes having counts of zero
    nonzero_counts_mask = np.zeros(num_classes, dtype=bool)
    nonzero_counts_mask[targets] = True
    class_counts = class_counts[:, nonzero_counts_mask]
    if stratify_how == 'balance':
        for b, counts in enumerate(class_counts):
            assert counts.max() - counts.min() <= 1
    elif stratify_how == 'match' and num_batches > 2:
        # every idx should be sampled exactly once
        idx_appeared = np.zeros(len(targets), dtype=bool)
        taken_idxs = batches.ravel()
        idx_appeared[taken_idxs] = 1
        if add_stragglers:
            idx_appeared[last_batch] = 1
        assert all(idx_appeared)
