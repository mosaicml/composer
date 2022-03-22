# Copyright 2021 MosaicML. All Rights Reserved.

"""A collection of common torchmetrics and loss functions."""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import to_categorical

__all__ = ["MIoU", "Dice", "CrossEntropyLoss", "soft_cross_entropy"]


class MIoU(Metric):
    """Torchmetrics mean Intersection-over-Union (mIoU) implementation.

    IoU calculates the intersection area between the predicted class mask and the label class mask.
    The intersection is then divided by the area of the union of the predicted and label masks.
    This measures the quality of predicted class mask with respect to the label. The IoU for each
    class is then averaged and the final result is the mIoU score. Implementation is primarily
    based on `mmsegmentation <https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/evaluation/metrics.py#L132>`_

    Args:
        num_classes (int): the number of classes in the segmentation task.
        ignore_index (int, optional): the index to ignore when computing mIoU. Default: ``-1``.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1):
        super().__init__(dist_sync_on_step=True)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state("total_intersect", default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total_union", default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, logits: Tensor, targets: Tensor):
        """Update the state with new predictions and targets."""
        preds = logits.argmax(dim=1)
        for pred, target in zip(preds, targets):
            mask = (target != self.ignore_index)
            pred = pred[mask]
            target = target[mask]

            intersect = pred[pred == target]
            area_intersect = torch.histc(intersect.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
            area_prediction = torch.histc(pred.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
            area_target = torch.histc(target.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

            self.total_intersect += area_intersect
            self.total_union += area_prediction + area_target - area_intersect

    def compute(self):
        """Aggregate state across all processes and compute final metric."""
        return 100 * (self.total_intersect / self.total_union).mean()  # type: ignore - / unsupported for Tensor|Module


class Dice(Metric):
    """The Dice Coefficient for evaluating image segmentation.

    The Dice Coefficient measures how similar predictions and targets are.
    More concretely, it is computed as 2 * the Area of Overlap divided by
    the total number of pixels in both images.

    Args:
        num_classes (int): the number of classes in the segmentation task.
    """

    def __init__(self, num_classes):
        super().__init__(dist_sync_on_step=True)
        self.add_state("n_updates", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((num_classes,)), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor):
        """Update the state based on new predictions and targets."""
        self.n_updates += 1  # type: ignore
        self.dice += self.compute_stats(pred, target)

    def compute(self):
        """Aggregate the state over all processes to compute the metric."""
        dice = 100 * self.dice / self.n_updates  # type: ignore
        best_sum_dice = dice[:]
        top_dice = round(torch.mean(best_sum_dice).item(), 2)
        return top_dice

    @staticmethod
    def compute_stats(pred: Tensor, target: Tensor):
        num_classes = pred.shape[1]
        scores = torch.zeros(num_classes - 1, device=pred.device, dtype=torch.float32)
        for i in range(1, num_classes):
            if (target != i).all():
                # no foreground class
                _, _pred = torch.max(pred, 1)
                scores[i - 1] += 1 if (_pred != i).all() else 0
                continue
            _tp, _fp, _tn, _fn, _ = _stat_scores(pred, target, class_index=i)  # type: ignore
            denom = (2 * _tp + _fp + _fn).to(torch.float)
            score_cls = (2 * _tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - 1] += score_cls
        return scores


def _stat_scores(
    preds: Tensor,
    target: Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

    if preds.ndim == target.ndim + 1:
        preds = to_categorical(preds, argmax_dim=argmax_dim)

    tp = ((preds == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((preds == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((preds != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((preds != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def _infer_target_type(input: Tensor, targets: Tensor) -> str:
    """Infers whether the target is in indices format or one_hot format.

    Example indices format: [1, 4, 7] Example one_hot format [[0, 1, 0], [1, 0, 0], ...]
    """
    if input.shape == targets.shape:
        return 'one_hot'
    elif input.ndim == targets.ndim + 1:
        return 'indices'
    else:
        raise RuntimeError(f'Unable to infer indices or one_hot. Targets has shape {targets.shape}'
                           f' and the inputs to cross entropy has shape {input.shape}. For one_hot, '
                           'expect targets.shape == inputs.shape. For indices, expect '
                           'inputs.ndim == targets.ndim + 1')


def ensure_targets_one_hot(input: Tensor, targets: Tensor, num_classes: Optional[float] = None) -> Tensor:
    r"""Ensures that the targets are in a one-hot format rather than an index format.

    Args:
        input (torch.Tensor): :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        target (torch.Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        num_classes (int, optional): Number of classes. If not specified, this will be inferred
            from input. Default: ``None``
    """
    if _infer_target_type(input, targets) == 'indices':
        # If the number of classes isn't specified, attempt to infer it from the input
        if num_classes is None:
            num_classes = input.shape[1]
        if targets.min() < 0:
            warnings.warn("Negative label indices are being ignored in conversion to one-hot labels")
            # Map all negative indicies to a class to drop.
            targets[targets < 0] = num_classes
            targets = F.one_hot(targets, num_classes=num_classes + 1)
            targets = torch.movedim(targets, -1, 1)
            # Drop any negative indices.
            targets = targets[:, 0:-1]
        else:
            targets = F.one_hot(targets, num_classes=num_classes)
            targets = torch.movedim(targets, -1, 1)
        targets = targets.float()
    return targets


def _check_for_index_targets(targets: Tensor) -> bool:
    """Checks if a given set of targets are indices by looking at the type."""
    index_types = ['torch.LongTensor', 'torch.cuda.LongTensor']
    return targets.type() in index_types


def soft_cross_entropy(input: Tensor,
                       target: Tensor,
                       weight: Optional[Tensor] = None,
                       size_average: Optional[bool] = None,
                       ignore_index: int = -100,
                       reduce: Optional[bool] = None,
                       reduction: str = 'mean'):
    r"""Drop-in replacement for :class:`~torch.nn.CrossEntropyLoss` that can handle class indices or one-hot labels.

    Args:
        input (torch.Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        target (torch.Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        weight (torch.Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`. Default: ``None``.
        size_average (bool, optional): Deprecated (see `reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field ``size_average``
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When ``size_average`` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            ``ignore_index`` is only applicable when the target contains class indices.
            Default: ``-100``
        reduce (bool, optional): Deprecated (see ``reduction``). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on `size_average`. When ``reduce`` is ``False``, returns a loss per
            batch element instead and ignores `size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: ``size_average``
            and ``reduce`` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override ``reduction``. Default: ``'mean'``

    This function will be obsolete with `this update <https://github.com/pytorch/pytorch/pull/61044>`_.
    """
    target_type = _infer_target_type(input, target)

    if target_type == 'indices':
        return F.cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction)
    elif target_type == 'one_hot':
        assert reduction in ['sum', 'mean', 'none'], f"{reduction} reduction not supported."
        assert size_average is None, "size_average is deprecated"
        assert reduce is None, "reduce is deprecated"
        if ignore_index != -100:
            warnings.warn("ignore_index not supported when using dense labels. Ignoring targets with 0 probability.")
        xentropy = -(target * F.log_softmax(input, dim=1))

        if weight is not None:
            # Ugly dimension shuffle to make multiplication work.
            xentropy = torch.movedim(xentropy, 1, -1)
            xentropy *= weight  # PyTorch doesn't normalize weights
            xentropy = torch.movedim(xentropy, -1, 1)

        xentropy = xentropy.sum(dim=1)
        num_examples = torch.numel(xentropy)

        if reduction == 'sum':
            xentropy = xentropy.sum()
        elif reduction == 'mean':
            xentropy = xentropy.mean()
            # Reweight loss to account for examples with less than 1 total probability (ignored examples)
            total_prob = target.sum()
            if total_prob <= 0:
                raise ValueError("No targets have nonzero probability")
            if total_prob < num_examples:
                warnings.warn("Some targets have less than 1 total probability.")
            xentropy *= num_examples / total_prob

        return xentropy
    else:
        raise ValueError(f"Unrecognized target type {target_type}")


class CrossEntropyLoss(Metric):
    """Torchmetrics cross entropy loss implementation.

    This class implements cross entropy loss as a :class:`torchmetrics.Metric` so that it can be returned by the
    :meth:`~.ComposerModel.metrics`.

    Args:
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. ``ignore_index`` is only applicable when the target
            contains class indices. Default: ``-100``.

        dist_sync_on_step (bool, optional): sync distributed metrics every step. Default: ``False``.
    """

    def __init__(self, ignore_index: int = -100, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.add_state("sum_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new predictions and targets."""
        # Loss calculated over samples/batch, accumulate loss over all batches
        self.sum_loss += soft_cross_entropy(preds, target, ignore_index=self.ignore_index)
        assert isinstance(self.total_batches, Tensor)
        self.total_batches += 1

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        assert isinstance(self.total_batches, Tensor)
        assert isinstance(self.sum_loss, Tensor)
        return self.sum_loss / self.total_batches
