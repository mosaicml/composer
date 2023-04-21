# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Label Smoothing classes and functions."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from composer import Evaluator
from composer.core import Algorithm, DataSpec, Event, State
from composer.loggers import Logger
from composer.loss.utils import ensure_targets_one_hot

__all__ = ['LabelSmoothing', 'smooth_labels']


def infer_num_classes(targets: torch.Tensor) -> int:
    """targets: Tensor containing index class labels; should not be one hot."""
    return int(targets.max() + 1)


class BaseSmoother:
    """All smoothers inherit from this class."""

    def __init__(self,
                 smoothing_type: str,
                 smoothing: float,
                 targets: Optional[torch.Tensor] = None,
                 num_classes: Optional[int] = None):
        if smoothing < 0 or smoothing > 1:
            raise ValueError('`smoothing` must be in range [0, 1]')
        if targets is None and num_classes is None:
            raise RuntimeError('`targets` and `num_classes` cannot both be `None`.')

        self.smoothing_type = smoothing_type
        self.smoothing = smoothing
        if num_classes is None:
            assert targets is not None
            num_classes = infer_num_classes(targets)
        self.num_classes = num_classes

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distributions = torch.empty((self.num_classes, self.num_classes))

    def get_hard_labels(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_soft_labels(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def smooth_labels(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = ensure_targets_one_hot(logits, targets)
        hard_labels = self.get_hard_labels(logits, targets)
        soft_labels = self.get_soft_labels(logits, targets)
        smoothed_labels = (1. - self.smoothing) * hard_labels + self.smoothing * soft_labels
        return smoothed_labels

    def update_batch(self, logits: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError

    def update_epoch(self):
        raise NotImplementedError

    def __repr__(self):
        return '<{}Smoother(smoothing={}) at {}>'.format(self.smoothing_type.capitalize(), self.smoothing,
                                                         hex(id(self)))

    def __str__(self):
        return '{}Smoother(smoothing={})'.format(self.smoothing_type.capitalize(), self.smoothing)


class UniformSmoother(BaseSmoother):
    """Smooths targets according to a uniform distribution as in `Szegedy et al <https://arxiv.org/abs/1512.00567>`_.

    The smoothed labels are computed as ``(1 - smoothing) * targets + smoothing * unif``
    where ``unif`` is a vector with elements all equal to ``1 / num_classes``.
    """

    def __init__(self,
                 smoothing: float = 0.1,
                 targets: Optional[torch.Tensor] = None,
                 num_classes: Optional[int] = None):
        super().__init__('uniform', smoothing, targets=targets, num_classes=num_classes)
        uniform = torch.tensor(1 / self.num_classes, device=self.device)
        self.distributions = uniform.expand((self.num_classes, self.num_classes))

    def get_hard_labels(self, logits: torch.Tensor, targets: torch.Tensor):
        return targets

    def get_soft_labels(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.distributions[0].expand(targets.shape).clone()

    def update_batch(self, logits: torch.Tensor, target: torch.Tensor):
        """UniformSmoother does not update the distribution in a batch"""
        pass

    def update_epoch(self):
        """UniformSmoother does not update the distribution in an epoch"""
        pass


class CategoricalSmoother(BaseSmoother):
    """Smooths targets according a categorical distribution. Recommended when
    classes are imbalanced.

    The smoothed labels are computed as ``(1 - smoothing) * targets + smoothing * cat``
    where ``cat`` is a vector where ``cat[c]`` is the percentage of datapoints
    that belongs to class ``c``, i.e ``cat[c] = 1/n \\sum_{i=1}^n 1[x_i == c]``
    where ``1[\\cdot]`` is an indicator function.
    """

    def __init__(self,
                 smoothing: float = 0.1,
                 targets: Optional[torch.Tensor] = None,
                 num_classes: Optional[int] = None):
        super().__init__('categorical', smoothing, targets=targets, num_classes=num_classes)

        if targets is not None:
            # Because `targets` may not include every class, we concatenate `all_classes` to ensures that we have at least one instance
            # of every class in the data. This guarantees that `counts` has the correct dimension but does not ultimately change `categorical`.
            all_classes = torch.arange(start=0, end=self.num_classes, step=1, device=self.device).unsqueeze(1)
            _, counts = torch.vstack((targets.unsqueeze(1), all_classes)).unique(return_counts=True)
            categorical = counts / counts.sum()
            self.distributions = categorical.expand((self.num_classes, self.num_classes))

    def get_hard_labels(self, logits: torch.Tensor, targets: torch.Tensor):
        return targets

    def get_soft_labels(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.distributions[0].expand(targets.shape).clone()

    def update_batch(self, logits: torch.Tensor, target: torch.Tensor):
        """CategoricalSmoother does not update the distribution in a batch"""
        pass

    def update_epoch(self):
        """CategoricalSmoother does not update the distribution in an epoch"""
        pass


class OnlineSmoother(BaseSmoother):
    """Smooths targets according an online distribution which gives higher
    probabilities to classes that are similar. Generally outperforms uniform
    smoothing. From `Zhang et al <https://arxiv.org/abs/2011.12562>`_.

    OnlineSmoother maintains an online (moving) label distribution for each class,
    which is updated after each training epoch. On sample `(x_i, y_i)` this
    distribution gives higher probabilities to classes that are more similar to
    the true class `y_i`. For example, in image classification when the true
    label is cat, the dog class will have greater probability than the car class
    because a dog looks more similar to a cat than a car does.

    The smoothed labels are computed as ``(1 - smoothing) * targets + smoothing * online``
    where ``online`` is a vector where the ``i``th entry is the probability of
    seeing the ``i``th class given the true label in ``targets``.

    The online distribution for the ``i``th class is updated to be the average
    probability distribution across all correctly predicted samples.
    Mathematically, the distribution of the ``i``th class is updated to be ``1 /
    n_i  \\sum_{j=1}^{n} 1[y_j == c] p_j`` where ``n_i`` is the number of samples
    correctly classified as being in class ``i``, ``n`` is the number of
    training samples in the current epoch, ``y_j`` the model's predicted class
    for the ``j``th sample, ``1[a]`` is an indicator vector which returns 1 if
    ``a`` is true and 0 otherwise, and ``p_j = softmax(f(x_j))`` is the predicted
    probability distributions over all the classes (i.e. softmax-ed logits)
    outputted by model ``f`` on the current sample ``x_j``.

    Our implementation differs from that of `Zhang et al` in several ways:
    1. In our implementation, the ``i``th row represents the probability
        distribution of the ``i``th class but in `Zhang et al` the columns
        represent the probability distribution.
    2. Our labels are computed as ``(1 - smoothing) * targets + smoothing *
        online`` whereas ``Zhang et al``'s labels are computed as ``alpha *
        targets + (1 - alpha) * online``. We have the ``1-`` on the ``targets``
        term and not on the ``online`` term because uniform label smoothing is
        generally implemented with the ``1-`` on the ``targets`` and we wish to
        maintain a consistent API.
    """

    def __init__(self,
                 smoothing: float = 0.1,
                 targets: Optional[torch.Tensor] = None,
                 num_classes: Optional[int] = None):
        super().__init__('online', smoothing, targets=targets, num_classes=num_classes)

        uniform = torch.tensor(1 / self.num_classes, device=self.device)
        self.distributions = uniform.expand((self.num_classes, self.num_classes))
        self.update_distributions = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        self.counts = torch.zeros(self.num_classes, device=self.device)

    def get_hard_labels(self, logits: torch.Tensor, targets: torch.Tensor):
        return targets

    def get_soft_labels(self, logits: torch.Tensor, targets: torch.Tensor):
        y_h = torch.argmax(logits, dim=1)
        soft_labels = torch.index_select(self.distributions, 0, y_h)
        return soft_labels

    def update_batch(self, logits: torch.Tensor, targets: torch.Tensor):
        """Once per batch, update ``update_distributions`` and ``counts``.

        When a sample ``(x_i, y_i)`` is correctly classified by the model, we add its predicted probability distribution to
        the ``y_i``th row of ``update_distributions`` and add one to the
        ``y_i``th element of ``counts``. We will use these quantities to update
        ``distributions`` at the end of the epoch.

        Mathematically, the update rules for a sample ``(x_i, y_i)`` are:
        1. ``online[y_i] += y_h_prob * 1[y_i == y_h]
        2. ``counts[y_i] += 1[y_i == y_h]``
        where ``y_h`` is the models predicted class, 1[x] is an indicator
        function on the boolean statement x, and ``y_h_prob`` is the model's
        ``logits`` after being softmaxed, thus forming a valid probability
        distribution predicted by the model.
        """
        with torch.no_grad():  # needed b/c logits.requires_grad=True

            # get predicted classes (y_h) and predicted probability distribution (y_h_prob) normalized via softmax
            targets = ensure_targets_one_hot(logits, targets)
            y_h = torch.argmax(logits, dim=1)
            y_h_prob = torch.softmax(logits, dim=1, dtype=torch.float32)

            # filter by correct class
            target_idxs = torch.nonzero(targets)[:, 1]
            mask = torch.eq(y_h, target_idxs)
            y_h_c = y_h[mask]
            y_h_prob_c = y_h_prob[mask]

            # update label distributions and count
            self.update_distributions = self.update_distributions.index_add(0, y_h_c, y_h_prob_c)
            self.counts = self.counts.index_add(0, y_h_c, torch.ones_like(y_h_c, dtype=torch.float32))

        return self.update_distributions, self.counts

    def update_epoch(self):
        """Once per epoch, update `distributions` and reset `update_distributions` and `counts`."""

        # normalize the update
        self.counts[torch.eq(self.counts, 0)] = 1  # avoid 0 denominator
        self.update_distributions = (self.update_distributions.T / self.counts).T

        # update values
        self.distributions = self.update_distributions.clone()
        self.update_distributions.zero_()
        self.counts.zero_()

        return self.distributions, self.update_distributions, self.counts


def smoothing_type_to_smoother(smoothing_type: str = 'uniform'):

    type_to_smoother = {'uniform': UniformSmoother, 'categorical': CategoricalSmoother, 'online': OnlineSmoother}

    if smoothing_type not in type_to_smoother:
        raise ValueError("'{}' is not a valid smoothing_type. Valid smoothing_type values are: {}.".format(
            smoothing_type, list(type_to_smoother.keys())))
    LabelSmoother = type_to_smoother[smoothing_type]
    return LabelSmoother


def smooth_labels(logits: torch.Tensor,
                  targets: torch.Tensor,
                  smoothing: float = 0.1,
                  smoothing_type: str = 'uniform',
                  num_classes: Optional[int] = None):
    """Smooth the labels with uniform, categorical, or online distributions."""

    LabelSmoother = smoothing_type_to_smoother(smoothing_type)
    label_smoother = LabelSmoother(smoothing=smoothing, targets=targets, num_classes=num_classes)
    smoothed_labels = label_smoother.smooth_labels(logits, targets)
    return smoothed_labels


class LabelSmoothing(Algorithm):

    def __init__(self,
                 smoothing: float = 0.1,
                 smoothing_type: str = 'uniform',
                 target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
                 save_distributions: bool = True):
        """Smooth the labels with uniform, categorical, or online distributions."""

        self.smoothing = smoothing
        self.smoothing_type = smoothing_type
        self.target_key = target_key
        self.label_smoother: Optional[Union[UniformSmoother, CategoricalSmoother, OnlineSmoother]] = None
        self.save_distributions = save_distributions
        self.original_labels = torch.Tensor()

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.FIT_START, Event.BEFORE_LOSS, Event.AFTER_LOSS, Event.EPOCH_END]

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:

        # initalize the label smoother
        if event == Event.FIT_START:
            err_msg = '`self.label_smoother` must be of type `UniformSmoother`, `CategoricalSmoother`, or `OnlineSmoother`, not `{}`.'.format(
                type(self.label_smoother))
            assert isinstance(self.label_smoother, Union[UniformSmoother, CategoricalSmoother, OnlineSmoother]), err_msg
            assert isinstance(state.train_dataloader, Union[Evaluator, DataLoader, DataSpec])
            
            targets = torch.tensor(state.train_dataloader.dataset.targets, device=state.device._device)
            LabelSmoother = smoothing_type_to_smoother(self.smoothing_type)
            self.label_smoother = LabelSmoother(smoothing=self.smoothing, targets=targets)
            if self.save_distributions:
                logger.log_metrics({'label_distributions': self.label_smoother.distributions})

        # compute smoothed labels and update label distributions (batch)
        if event == Event.BEFORE_LOSS:
            labels = state.batch_get_item(self.target_key)
            self.original_labels = labels.clone()

            assert isinstance(state.outputs, torch.Tensor), 'Multiple tensors not supported yet'
            assert isinstance(labels, torch.Tensor), 'Multiple tensors not supported yet'
            err_msg = '`self.label_smoother` must be of type `UniformSmoother`, `CategoricalSmoother`, or `OnlineSmoother`, not {}.'.format(
                type(self.label_smoother))
            assert isinstance(self.label_smoother, Union[UniformSmoother, CategoricalSmoother, OnlineSmoother]), err_msg

            smoothed_labels = self.label_smoother.smooth_labels(state.outputs, labels)
            state.batch_set_item(self.target_key, smoothed_labels)

            self.label_smoother.update_batch(state.outputs, labels)

        # restore the target to the non-smoothed version
        elif event == Event.AFTER_LOSS:
            state.batch_set_item(self.target_key, self.original_labels)

        # update label distrbutions (epoch)
        elif event == Event.EPOCH_END:
            err_msg = '`self.label_smoother` must be of type `UniformSmoother`, `CategoricalSmoother`, or `OnlineSmoother`, not `{}`.'.format(
                type(self.label_smoother))
            assert isinstance(self.label_smoother, Union[UniformSmoother, CategoricalSmoother, OnlineSmoother]), err_msg
            
            self.label_smoother.update_epoch()
