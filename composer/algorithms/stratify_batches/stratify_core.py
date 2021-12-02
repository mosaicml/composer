# Copyright 2021 MosaicML. All Rights Reserved.

import math
from typing import Iterable, List

import numpy as np
from torch.utils.data import Sampler

# TODO refactor all of this to use rng = default_rng(), rng.whatever() instead of np.random.whatever()


def _groupby_ints(targets: Iterable):
    """Given a sequence of integers, returns a dict mapping unique integers to
    lists of indices at which they occur. E.g.,

    >>> targets = [1, 3, 2, 3, 1, 2]
    >>> _groupby_ints(targets)
    {1: array([0, 4]), 2: array([2, 5]), 3: array([1, 3])}
    """
    targets = np.asarray(targets)
    assert len(targets.shape) == 1, "Targets must be a vector of integer class labels"
    uniqs, counts = np.unique(targets, return_counts=True)
    idxs_for_uniqs = np.split(np.argsort(targets), indices_or_sections=np.cumsum(counts))
    return dict(zip(uniqs, idxs_for_uniqs))


class BalancedSampler:
    """Enforces that number of times any two elements have been sampled
    differs by at most one.

    Within a single pass through the data, this means that no element is
    sampled twice before all elements have been sampled at least once.
    """

    def __init__(self, data: np.array, replace=False):
        self.data = data.copy()
        # self.replace = replace
        self.reset()

    def reset(self) -> None:
        np.random.shuffle(self.data)
        self.tail = self.data
        self.num_epochs_completed = 0

    def is_reset(self):
        return (self.num_epochs_completed == 0) and len(self.tail) == len(self.data)

    def __len__(self):
        return len(self.data)

    def sample(self, count: int) -> List:
        if count < 1:
            return []

        n = len(self.data)
        ret = []
        num_copies = count // n
        if num_copies > 0:
            # so many elems requested we need to feed it whole array one or more times
            copy_idxs = np.random.permutation(n * num_copies) % n
            ret += list(self.data[copy_idxs])
            count -= n * num_copies
            self.num_epochs_completed += num_copies
        if count >= len(self.tail):
            # pass in the whole tail and re-init the shuffled array
            # if count > len(self.tail) and not self.replace:
            #     raise IndexError(f"Tried to {count} elements when only {len(self.tail)} remained!")
            ret += list(self.tail)
            count -= len(self.tail)
            # self.tail = self.tail[-1:-1]   # empty array
            self.num_epochs_completed += 1
            # if self.replace:
            np.random.shuffle(self.data)
            self.tail = self.data
        if count > 0:
            # simple case: pass it next few elems in our permuted array
            ret += list(self.tail[:count])
            self.tail = self.tail[count:]
        return ret

    def _num_elements_unsampled_at_epoch(self, epoch: int):
        if epoch < self.num_epochs_completed:
            return 0
        if epoch > self.num_epochs_completed:
            return len(self.data)
        # at current epoch, tail is the collection of unsampled elements
        return len(self.tail)

    def num_unsampled_elements(self):
        return self._num_elements_unsampled_at_epoch(0)


def _sample_batches_stratified(samplers: List[BalancedSampler], batch_size: int, num_batches: int) -> List:
    batches = []
    num_classes = len(samplers)
    class_idxs = np.arange(num_classes)
    for b in range(num_batches):
        batch = []
        num_remaining_batches = num_batches - b
        remaining_batch_size = batch_size
        # if enough samples in a class to take m >= 1 of them for each remaining batch, always
        # take m of them before any random sampling happens
        for sampler in samplers:
            take_num = sampler.num_unsampled_elements() // num_remaining_batches
            batch += sampler.sample(take_num)
            remaining_batch_size -= take_num
            assert remaining_batch_size >= 0, "BUG: we made a math error in stratified batch construction"
        # now randomly sample from elems that won't evenly fit into a batch
        straggler_counts = [s.num_unsampled_elements() % num_remaining_batches for s in samplers]
        probs = straggler_counts / straggler_counts.sum()
        replace = remaining_batch_size > num_classes
        use_classes = np.random.choice(class_idxs, replace=replace, p=probs, size=remaining_batch_size)
        for c in use_classes:
            batch += samplers[c].sample(1)
        batches.append(batch)
    return batches


def _sample_batches_balanced(samplers: List[BalancedSampler], batch_size: int, num_batches: int) -> List:
    batches = []
    num_classes = len(samplers)
    class_idxs = np.arange(len(num_classes))
    for sampler in samplers:
        if not sampler.is_reset():
            sampler.reset()
        sampler.replace = True  # enable sampling points more than once without throwing

    for _ in range(num_batches):
        batch = []
        # first take even number of elems from each class as much as possible before any randomness
        count_per_class = batch_size // num_classes
        for sampler in samplers:
            batch += sampler.sample(count_per_class)
        remaining_batch_size = batch_size % num_classes
        assert remaining_batch_size == batch_size - (count_per_class * num_classes)
        # sample one elem each from subset of classes, chosen uniformly at random
        use_classes = np.random.choice(class_idxs, size=remaining_batch_size, replace=False)
        for sampler in samplers[use_classes]:
            batch += sampler.sample(1)
        batches.append(batch)

    return batches


def _sample_stragglers(samplers: List[BalancedSampler], batch_size: int, total_num_samples: int) -> List:
    remaining_batch_size = batch_size
    batch = []
    shuffled_samplers = np.random.permutation(samplers)  # shuffle a copy
    for sampler in shuffled_samplers:
        idxs = sampler.sample(sampler.num_unsampled_elements())
        batch += idxs
        remaining_batch_size -= len(idxs)
        if remaining_batch_size <= 0:
            return batch[:batch_size]
    # if we got to here, we didn't have enough unsampled elements, so we'll have to sample some old ones;
    # since this only happens for final batch of epoch, keep things simple and sample uniformly at random
    batch += list(np.random.choice(np.arange(total_num_samples), size=remaining_batch_size), replace=False)
    return batch


class StratifiedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size, targets, shuffle=True, drop_last=False, stratify_how='balance'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.targets = targets  # TODO figure these out in 1st epoch if not provided
        self.shuffle = shuffle
        self.drop_last = drop_last
        allowed_stratify_hows = ('balance', 'match')
        if stratify_how not in allowed_stratify_hows:
            raise ValueError(f"stratify_how must be one of {allowed_stratify_hows}, not {stratify_how}")
        self.stratify_how = stratify_how
        self._set_targets(targets)

    def _set_targets(self, targets: Iterable):
        _class_to_idxs = _groupby_ints(targets)
        self.classes = list(_class_to_idxs.keys())
        self.num_classes = len(self.classes)
        self.samplers = [BalancedSampler(_class_to_idxs[c]) for c in self.classes]

    def __iter__(self):
        batches = []
        total_num_samples = len(self.dataset)
        num_batches = total_num_samples / self.num_classes
        f_round = math.floor if self.drop_last else math.ceil
        num_batches_total = int(f_round(num_batches))
        num_batches_no_stragglers = int(math.floor(num_batches))
        self._reset_samplers()  # reset sampling for each epoch

        if self.stratify_how == 'match':  # each batch mirrors overall distro on best-effort basis
            batches = _sample_batches_stratified(self.samplers, batch_size=self.batch_size, num_batches=num_batches_no_stragglers)
        elif self.stratify_how == 'balance':  # per-class sample counts vary by at most 1
            batches = _sample_batches_balanced(self.samplers, batch_size=self.batch_size, num_batches=num_batches_no_stragglers)

        if num_batches_total > num_batches_no_stragglers:
            batches.append(_sample_stragglers(self.samplers, batch_size=self.batch_sizee, total_num_samples=total_num_samples))
        return batches

    def __len__(self):
        return int(math.ceil(len(self.dataset) / self.batch_size))
