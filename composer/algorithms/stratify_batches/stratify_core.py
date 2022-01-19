# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike
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
        self.original_data = data.copy()  # decoupled for determinism after reset
        self.data = data.copy()
        self.replace = replace
        self.rng = rng or np.random.default_rng()
        self.reset()
        # redundant logic, but fixes pyright errors
        self.tail = self.data
        self.num_epochs_completed = 0

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        self.data = self.original_data.copy()
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


def _sample_batches_match_old(samplers: Sequence[BalancedSampler], batch_size: int, num_batches: int,
                              rng: np.random.Generator, **_) -> List:
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
        if remaining_batch_size > 0:
            tail_counts = np.array([s.num_unsampled_elements() for s in samplers])
            probs = tail_counts / tail_counts.sum()
            use_classes = rng.choice(num_classes, replace=False, p=probs, size=remaining_batch_size)
            for c in use_classes:
                batch += samplers[c].sample(1)

        batches.append(batch)
    return batches


def _sample_batches_uniform(samplers: Sequence[BalancedSampler], batch_size: Union[int, Sequence], num_batches: int,
                            rng: np.random.Generator, **_) -> List:
    unsampled_counts = [s.num_unsampled_elements() for s in samplers]
    # unsampled_counts = [len(s.tail) for s in samplers]
    total_unsampled_count = sum(unsampled_counts)

    batch_sizes = batch_size
    if isinstance(batch_size, (int, float)):
        batch_sizes = np.full(shape=num_batches, fill_value=batch_size)
    batch_sizes = np.asarray(batch_sizes)
    requested_count = batch_sizes.sum()

    if total_unsampled_count < requested_count:
        raise ValueError(f"Can't sample {requested_count} unique elements from a set of size {total_unsampled_count}")

    # choose how many elems to take from each sampler
    labels = np.empty(total_unsampled_count, dtype=np.int)
    label_write_ptr = labels
    for c in range(len(samplers)):
        count = unsampled_counts[c]
        label_write_ptr[:count] = c
        label_write_ptr = label_write_ptr[count:]
    shuffled_labels = rng.permutation(labels)[:requested_count]
    uniqs, counts = np.unique(shuffled_labels, return_counts=True)

    # now sample the correct number of elems from each sampler
    flat_batches = np.empty(requested_count, dtype=np.int)
    write_ptr = flat_batches
    for c, count in zip(uniqs, counts):
        write_ptr[:count] = samplers[c].sample(count)
        write_ptr = write_ptr[count:]
    flat_batches = rng.permutation(flat_batches)

    # pull out correct number of samples for each batch; we allow variable
    # numbers so that this function can be used to handle straggler sampling
    # within other sampling functions
    batches = []
    for b in range(num_batches):
        batch_size = batch_sizes[b]
        batches.append(list(flat_batches[:batch_size]))
        flat_batches = flat_batches[batch_size:]
    return batches


def _sample_batches_match(samplers: Sequence[BalancedSampler], batch_size: int, num_batches: int,
                          rng: np.random.Generator, **_) -> List:
    batches = [[] for _ in range(num_batches)]
    num_classes = len(samplers)
    # evenly spread samples from a given class across batches as much as possible
    for sampler in samplers:
        take_num = sampler.num_unsampled_elements() // num_batches
        take_num = min(take_num, batch_size // num_classes)  # requesting few batches
        for batch in batches:
            batch += sampler.sample(take_num)

    tail_size = batch_size - len(batches[0])  # same for all batches
    if tail_size == 0:
        return batches
    assert tail_size > 0, f"tail size = {tail_size}"

    # sample uniformly at random for remaining entries
    tail_batches = _sample_batches_uniform(samplers=samplers, batch_size=tail_size, num_batches=num_batches, rng=rng)

    return [batches[b] + tail_batches[b] for b in range(num_batches)]


def _sample_batches_balanced(samplers: Sequence[BalancedSampler], batch_size: int, num_batches: int,
                             rng: np.random.Generator, **_) -> List:
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


def _compute_p_proportional(samplers):
    # prob of each class is proportional to its number of unsampled
    # elements; this results in uniform sampling of elements
    unsampled_counts = np.array([s.num_unsampled_elements() for s in samplers])
    return unsampled_counts / (unsampled_counts.sum() + 1e-10)


def _sample_batches_imbalanced(samplers: Sequence[BalancedSampler], batch_size: int, num_batches: int,
                               rng: np.random.Generator, imbalance: float, **_) -> List:
    assert 0. <= imbalance <= 1.
    batches = []
    num_classes = len(samplers)
    taken_counts = np.zeros(num_classes)

    for _ in range(num_batches):
        batch = []
        taken_counts[:] = 0
        p_proportional = _compute_p_proportional(samplers)
        take_class = rng.choice(num_classes, size=1, p=p_proportional)[0]
        taken_counts[take_class] = 1
        batch.append(samplers[take_class].sample(1)[0])
        for _ in range(batch_size - 1):
            p_proportional = _compute_p_proportional(samplers)
            # handle a previously taken class having no more new samples
            p_self_excite = (taken_counts + 1e-10) * (p_proportional > 0)
            # sample based on convex combo of uniform and self-exiciting distro
            p_self_excite /= p_self_excite.sum()
            probs = imbalance * p_self_excite + (1. - imbalance) * p_proportional
            take_class = rng.choice(num_classes, size=1, p=probs)[0]
            taken_counts[take_class] += 1
            batch.append(samplers[take_class].sample(1)[0])
        batches.append(batch)

    return batches


# experimental control that should be equivalent to uniform sampling without replacement
def _sample_batches_naive_uniform(samplers: Sequence[BalancedSampler], batch_size: int, num_batches: int,
                                  rng: np.random.Generator, **_) -> List:
    batches = []
    num_classes = len(samplers)
    for _ in range(num_batches):
        batch = []
        for _ in range(batch_size):
            p_proportional = _compute_p_proportional(samplers)
            take_class = rng.choice(num_classes, size=1, p=p_proportional)[0]
            batch.append(samplers[take_class].sample(1)[0])
        batches.append(batch)
    return batches


def _sample_batches_echo_classes(samplers: Sequence[BalancedSampler],
                                 batch_size: int,
                                 num_batches: int,
                                 rng: np.random.Generator,
                                 targets: Sequence,
                                 echo_classes_factor: int = 2,
                                 stratify_how: str = 'uniform',
                                 **kwargs) -> List:
    num_classes = len(samplers)
    initial_batch_size = batch_size // echo_classes_factor
    if initial_batch_size < 1:
        raise ValueError(f"Total batch size {batch_size} does not exceed echo factor {echo_classes_factor}")
    if stratify_how == 'echo_classes':
        raise ValueError(f"Cannot use {stratify_how} as inner sampler for {stratify_how}")

    # construct initial batches based on inner sampling function
    f_sample = _NAME_TO_SAMPLING_FUNC[stratify_how]
    batches = f_sample(samplers=samplers, batch_size=initial_batch_size, num_batches=num_batches, rng=rng, **kwargs)

    if echo_classes_factor < 2:  # no echoing, so initial sampling suffices
        return batches

    # now take more samples in each batch with the same distribution as the
    # batch-so-far; if any classes run out of samples, take remaining samples
    # from whichever sampler(s) have the most unsampled elems
    flat_batches = np.array(batches).ravel()
    flat_labels = targets[flat_batches]
    labels = flat_labels.reshape(num_batches, initial_batch_size)
    unsampled_counts = np.array([s.num_unsampled_elements() for s in samplers])
    for batch, batch_labels in zip(batches, labels):
        class_counts = np.bincount(batch_labels, minlength=num_classes)
        take_nums = class_counts * (echo_classes_factor - 1)
        take_nums = np.minimum(take_nums, unsampled_counts)
        nonzero_classes = np.where(take_nums > 0)[0]
        for c in nonzero_classes:
            sampler = samplers[c]
            take_num = take_nums[c]
            batch += sampler.sample(take_num)
            unsampled_counts[c] -= take_num
        # permute at this point so that samples from a given class aren't
        # all consecutive; this would result in higher-number replicas loading
        # batches that consist of only a few classes, which might mess up the
        # batchnorms, or at least introduce ordering as a lurking variable when
        # comparing this approach to echoing individual samples
        batch = list(rng.permutation(batch))

    # finish off batches with uniform sampling; batches end up incomplete
    # if any of their classes run out of samples
    tail_sizes = np.array([batch_size - len(batch) for batch in batches])
    assert (tail_sizes >= 0).all(), "Something is wrong with class echoing logic!"
    # tail_sizes = np.maximum(tail_sizes, 0)
    batch_tails = _sample_batches_uniform(samplers=samplers, batch_size=tail_sizes, num_batches=num_batches, rng=rng)
    return [head + tail for head, tail in zip(batches, batch_tails)]


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


def create_samplers(targets: Any, replace: bool) -> Tuple[ArrayLike, ArrayLike]:
    _class_to_idxs = groupby_ints(targets)
    classes = list(_class_to_idxs.keys())
    samplers = [BalancedSampler(_class_to_idxs[c], replace=replace) for c in classes]
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


_NAME_TO_SAMPLING_FUNC = {
    'balance': _sample_batches_balanced,
    'match': _sample_batches_match,
    'imbalance': _sample_batches_imbalanced,
    'uniform': _sample_batches_uniform,
    'naive_uniform': _sample_batches_naive_uniform,
    'echo_classes': _sample_batches_echo_classes,
}

T_co = TypeVar('T_co', covariant=True)


class StratifiedBatchSampler(DistributedSampler[T_co]):

    def __init__(self,
                 dataset: Sequence,
                 *,
                 batch_size: int,
                 targets: Optional[Sequence] = None,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 stratify_how: str = 'match',
                 echo_samples_factor: int = 1,
                 echo_classes_factor: int = 1,
                 imbalance: float = .1,
                 targets_attr: Optional[str] = None,
                 seed: int = 42,
                 **kwargs):
        super().__init__(dataset=dataset, shuffle=shuffle, drop_last=drop_last, seed=seed, **kwargs)
        self.batch_size = batch_size
        if targets is None:
            targets = extract_targets_from_dataset(dataset, targets_attr=targets_attr)

        # map original targets to ints 0 thru len(unique(targets)); this is okay
        # because we only ever return sample indices, not their associated labels
        raw_targets = np.asarray(targets).ravel()
        _, self.targets = np.unique(raw_targets, return_inverse=True)

        self.shuffle = shuffle
        if not shuffle:
            raise NotImplementedError("Sampling with shuffle=False is not yet supported")
        self.drop_last = drop_last
        allowed_stratify_hows = _NAME_TO_SAMPLING_FUNC.keys()
        if stratify_how not in allowed_stratify_hows:
            raise ValueError(f"stratify_how must be one of {allowed_stratify_hows}, not {stratify_how}")
        self.stratify_how = stratify_how
        self.echo_samples_factor = int(echo_samples_factor)
        self.echo_classes_factor = int(echo_classes_factor)
        self.imbalance = imbalance
        self.total_batch_size = batch_size * self.num_replicas
        self._set_targets(targets)

        self.total_num_samples = len(self.targets)
        self.num_full_batches = self.total_num_samples // self.total_batch_size
        has_stragglers = self.num_full_batches * self.total_batch_size < self.total_num_samples
        self.add_stragglers = has_stragglers and not self.drop_last

        # our logic differs from DistributedSampler, so make sure that
        # these public attributes reflect what's actually going on
        self.num_samples = self.batch_size * self.num_full_batches
        self.total_size = self.num_samples * self.num_replicas

    def _set_targets(self, targets: Iterable):
        replace = self.stratify_how == 'balance'
        self.samplers, self.classes = create_samplers(targets, replace=replace)
        self.num_classes = len(self.classes)

    def _reset_samplers(self, rng: np.random.Generator) -> None:
        for sampler in self.samplers:
            sampler.reset(rng)

    def __iter__(self) -> Iterator[T_co]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self._reset_samplers(rng)  # reset sampling for each epoch

        sample_total_batch_size = self.total_batch_size // self.echo_samples_factor
        # echo_classes sampling func calls other sampling funcs
        batches = _sample_batches_echo_classes(
            self.samplers,
            batch_size=sample_total_batch_size,
            num_batches=self.num_full_batches,
            rng=rng,
            imbalance=self.imbalance,
            echo_classes_factor=self.echo_classes_factor,
            targets=self.targets,
            stratify_how=self.stratify_how,
        )

        # make multiple copies of each batch if requested; important to append
        # them one after another so that a given idx tends to get spread across
        # replicas, instead of showing up many times for one replica;
        # apparently matters for batchnorms
        if self.echo_samples_factor > 1:
            new_batches = []
            for batch in batches:
                new_batch = batch[:]
                for _ in range(self.echo_samples_factor - 1):
                    new_batch += batch
                new_batches.append(new_batch)
            tail_sizes = [self.total_batch_size - len(batch) for batch in new_batches]
            batch_tails = _sample_batches_uniform(
                self.samplers,
                batch_size=tail_sizes,
                num_batches=len(batches),
                rng=rng,
            )
            batches = [head + tail for head, tail in zip(new_batches, batch_tails)]

        if self.add_stragglers:
            batches.append(
                sample_stragglers(self.samplers,
                                  batch_size=sample_total_batch_size,
                                  total_num_samples=self.total_num_samples,
                                  rng=rng))

        # shard each batch across the workers. We can't allocate whole batches
        # across them since these batches use the logical batch size, not the
        # per-worker batch size. We can't use the per-worker batch size because
        # this is smaller and can make balancing them meaningless,
        # if the class count is greater than the per-worker batch size.
        my_start_idx = self.rank * self.batch_size
        my_end_idx = my_start_idx + self.batch_size
        my_batches = [b[my_start_idx:my_end_idx] for b in batches]

        return iter(my_batches)

    def __len__(self):
        # batch sampler needs to return number of batches, not number
        # of samples; see https://github.com/pytorch/pytorch/blob/fca8a0acaa5b058249324ca399e22d29fda25722/torch/utils/data/sampler.py#L237 # noqa
        return self.num_full_batches + (1 if self.add_stragglers else 0)
