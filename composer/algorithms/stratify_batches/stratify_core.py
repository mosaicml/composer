# Copyright 2021 MosaicML. All Rights Reserved.

import math
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from torch.utils.data import DistributedSampler


def groupby_ints(targets: Iterable):
    """Given a sequence of integers, returns a dict mapping unique integers to
    lists of indices at which they occur. E.g.,

    >>> targets = [1, 3, 2, 3, 1, 2]
    >>> groupby_ints(targets)
    {1: array([0, 4]), 2: array([2, 5]), 3: array([1, 3])}
    """
    targets = np.asarray(targets)
    if targets.size > 1:
        targets = targets.squeeze()

    assert targets.ndim == 1, "Targets must be a vector of integer class labels"
    uniqs, counts = np.unique(targets, return_counts=True)
    idxs_for_uniqs = np.split(np.argsort(targets), indices_or_sections=np.cumsum(counts))
    return dict(zip(uniqs, idxs_for_uniqs))


class BalancedSampler:
    """Enforces that number of times any two elements have been sampled
    differs by at most one.

    Within a single pass through the data, this means that no element is
    sampled twice before all elements have been sampled at least once.
    """

    def __init__(self, data: np.array, replace: bool = False, rng: Optional[np.random.Generator] = None):
        self.data = data.copy()
        self.replace = replace
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.rng.shuffle(self.data)
        self.tail = self.data
        self.num_epochs_completed = 0

    def is_reset(self):
        return (self.num_epochs_completed == 0) and len(self.tail) == len(self.data)

    def __len__(self):
        return len(self.data)

    def sample(self, count: int) -> List:
        if count < 1:
            return []
        num_unsampled = self.num_unsampled_elements()
        if count > num_unsampled and not self.replace:
            raise ValueError(
                f"Cannot sample {count} elements when 'replace=False' and only {num_unsampled} unsampled elements remain"
            )

        n = len(self.data)
        ret = []
        num_copies = count // n
        if num_copies > 0:
            # so many elems requested we need to feed it whole array one or more times
            copy_idxs = self.rng.permutation(n * num_copies) % n
            ret += list(self.data[copy_idxs])
            count -= n * num_copies
            self.num_epochs_completed += num_copies
        if count >= len(self.tail):
            # pass in the whole tail and re-init the shuffled array
            ret += list(self.tail)
            count -= len(self.tail)
            self.num_epochs_completed += 1
            self.rng.shuffle(self.data)
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


def _sample_batches_match(samplers: Sequence[BalancedSampler], batch_size: int, num_batches: int,
                          rng: np.random.Generator) -> List:
    batches = []
    num_classes = len(samplers)
    for b in range(num_batches):
        batch = []
        num_remaining_batches = num_batches - b
        remaining_batch_size = batch_size
        # if enough samples in a class to take m >= 1 of them for each
        # remaining batch, always take m of them before any random sampling
        for sampler in samplers:
            take_num = sampler.num_unsampled_elements() // num_remaining_batches
            take_num = min(take_num, remaining_batch_size)
            batch += sampler.sample(take_num)
            remaining_batch_size -= take_num

        # now randomly sample from elems that won't evenly fit into a batch
        straggler_counts = np.array([s.num_unsampled_elements() % num_remaining_batches for s in samplers])
        # print("straggler counts: ", straggler_counts)
        num_stragglers = straggler_counts.sum()
        if num_stragglers > 0:
            probs = straggler_counts / num_stragglers
            straggler_take_num = min(remaining_batch_size, sum(probs > 0))
            use_classes = rng.choice(num_classes, replace=False, p=probs, size=straggler_take_num)
            for c in use_classes:
                batch += samplers[c].sample(1)
            remaining_batch_size -= straggler_take_num

        # possible that there weren't enough stragglers; this can happen if the
        # deterministic sampling turns the num_unsampled counts to be even
        # multiples of the remaining batch size for enough samplers
        tail_counts = np.array([s.num_unsampled_elements() for s in samplers])
        probs = tail_counts / tail_counts.sum()
        if remaining_batch_size > 0:
            use_classes = rng.choice(num_classes, replace=False, p=probs, size=remaining_batch_size)
            for c in use_classes:
                batch += samplers[c].sample(1)

        batches.append(batch)
    return batches


def _sample_batches_balanced(samplers: Sequence[BalancedSampler], batch_size: int, num_batches: int,
                             rng: np.random.Generator) -> List:
    batches = []
    num_classes = len(samplers)
    for _ in range(num_batches):
        batch = []
        # first take even number of elems from each class as much as
        # possible before any randomness
        count_per_class = batch_size // num_classes
        for sampler in samplers:
            batch += sampler.sample(count_per_class)
        remaining_batch_size = batch_size % num_classes
        assert remaining_batch_size == batch_size - (count_per_class * num_classes)
        # sample one elem each from subset of classes, chosen uniformly at random
        use_classes = rng.choice(num_classes, size=remaining_batch_size, replace=False)
        for sampler in samplers[use_classes]:
            batch += sampler.sample(1)
        batches.append(batch)

    return batches


def sample_stragglers(samplers: Sequence[BalancedSampler],
                      batch_size: int,
                      total_num_samples: int,
                      rng: np.random.Generator,
                      full_batch=True) -> List:
    batch = []
    remaining_batch_size = batch_size
    shuffled_samplers = rng.permutation(samplers)  # shuffle a copy
    for sampler in shuffled_samplers:
        idxs = sampler.sample(sampler.num_unsampled_elements())
        batch += idxs
        remaining_batch_size -= len(idxs)
        if remaining_batch_size <= 0:
            return batch[:batch_size]
    if not full_batch:
        return batch

    # if we got to here, we didn't have enough unsampled elements, so
    # we'll have to sample some old ones; since this only happens for
    # final batch of epoch, keep things simple and sample uniformly at random
    batch += list(rng.choice(total_num_samples, size=remaining_batch_size, replace=False))
    return batch


def create_samplers(targets: Sequence[int], replace: bool, rng: np.random.Generator) -> Tuple[np.array, np.array]:
    _class_to_idxs = groupby_ints(targets)
    classes = list(_class_to_idxs.keys())
    samplers = [BalancedSampler(_class_to_idxs[c], replace=replace, rng=rng) for c in classes]
    return np.array(samplers), np.array(classes)


def extract_targets_from_dataset(dataset: Sequence, targets_attr: Optional[str] = None):
    if targets_attr:
        targets = getattr(dataset, targets_attr)
    # torchvision DatasetFolder subclasses use 'targets'; some torchvision
    # datasets, like caltech101, use 'y' instead
    elif hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'y'):
        targets = dataset.y
    else:
        raise AttributeError("Since neither `targets` nor `targets_attr` "
                             "were provided, DataLoader.dataset must have an integer vector attribute "
                             "named either 'targets' or 'y'.")
    return targets


T_co = TypeVar('T_co', covariant=True)


class StratifiedBatchSampler(DistributedSampler[T_co]):

    def __init__(self,
                 dataset: Sequence,
                 *,
                 batch_size: int,
                 targets: Optional[Sequence] = None,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 stratify_how: str = 'balance',
                 targets_attr: Optional[str] = None,
                 seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None,
                 **kwargs):
        super().__init__(dataset=dataset, shuffle=shuffle, drop_last=drop_last, seed=seed, **kwargs)
        self.batch_size = batch_size

        if targets is None:
            targets = extract_targets_from_dataset(dataset, targets_attr=targets_attr)

        self.targets = np.asarray(targets).ravel()
        self.shuffle = shuffle
        if not shuffle:
            raise NotImplementedError("Sampling with shuffle=False is not yet supported")
        self.drop_last = drop_last
        allowed_stratify_hows = ('balance', 'match')
        if stratify_how not in allowed_stratify_hows:
            raise ValueError(f"stratify_how must be one of {allowed_stratify_hows}, not {stratify_how}")
        self.stratify_how = stratify_how
        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng = rng
        self._set_targets(targets)

    def _set_targets(self, targets: Iterable):
        replace = self.stratify_how == 'balance'
        self.samplers, self.classes = create_samplers(targets, replace=replace, rng=self.rng)
        self.num_classes = len(self.classes)

    def _reset_samplers(self):
        for sampler in self.samplers:
            sampler.reset()

    def __iter__(self) -> Iterator[T_co]:
        total_num_samples = len(self.targets)
        num_batches = total_num_samples // self.batch_size
        has_stragglers = num_batches * self.batch_size < total_num_samples
        self._reset_samplers()  # reset sampling for each epoch

        f_sample = {'balance': _sample_batches_balanced, 'match': _sample_batches_match}[self.stratify_how]
        batches = f_sample(self.samplers, batch_size=self.batch_size, num_batches=num_batches, rng=self.rng)

        if has_stragglers and not self.drop_last:
            batches.append(
                sample_stragglers(self.samplers,
                                  batch_size=self.batch_size,
                                  total_num_samples=total_num_samples,
                                  rng=self.rng))
        return iter(batches)

    def __len__(self):
        return int(math.ceil(len(self.dataset) / self.batch_size))
