# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""
from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import to_categorical

from composer.loss import soft_cross_entropy

__all__ = ['MIoU', 'Dice', 'CrossEntropy', 'LossMetric']


class MIoU(Metric):
    """Torchmetrics mean Intersection-over-Union (mIoU) implementation.

    IoU calculates the intersection area between the predicted class mask and the label class mask.
    The intersection is then divided by the area of the union of the predicted and label masks.
    This measures the quality of predicted class mask with respect to the label. The IoU for each
    class is then averaged and the final result is the mIoU score. Implementation is primarily
    based on `mmsegmentation <https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/core/evaluation/metrics.py#L132>`_

    Args:
        num_classes (int): the number of classes in the segmentation task.
        ignore_index (int, optional): the index to ignore when computing mIoU. Default: ``-1``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: int = -1):
        super().__init__(dist_sync_on_step=True)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state('total_intersect', default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('total_union', default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx='sum')

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
        total_intersect = self.total_intersect[self.total_union != 0]  # type: ignore (third-party)
        total_union = self.total_union[self.total_union != 0]  # type: ignore (third-party)
        return 100 * (total_intersect / total_union).mean()


class Dice(Metric):
    """The Dice Coefficient for evaluating image segmentation.

    The Dice Coefficient measures how similar predictions and targets are.
    More concretely, it is computed as 2 * the Area of Overlap divided by
    the total number of pixels in both images.

    Args:
        num_classes (int): the number of classes in the segmentation task.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, num_classes: int):
        super().__init__(dist_sync_on_step=True)
        self.add_state('n_updates', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('dice', default=torch.zeros((num_classes,)), dist_reduce_fx='sum')

    def update(self, preds: Tensor, targets: Tensor):
        """Update the state based on new predictions and targets."""
        self.n_updates += 1  # type: ignore
        self.dice += self.compute_stats(preds, targets)

    def compute(self):
        """Aggregate the state over all processes to compute the metric."""
        dice = 100 * self.dice / self.n_updates  # type: ignore
        best_sum_dice = dice[:]
        top_dice = round(torch.mean(best_sum_dice).item(), 2)
        return top_dice

    @staticmethod
    def compute_stats(preds: Tensor, targets: Tensor):
        num_classes = preds.shape[1]
        scores = torch.zeros(num_classes - 1, device=preds.device, dtype=torch.float32)
        for i in range(1, num_classes):
            if (targets != i).all():
                # no foreground class
                _, _pred = torch.max(preds, 1)
                scores[i - 1] += 1 if (_pred != i).all() else 0
                continue
            _tp, _fp, _tn, _fn, _ = _stat_scores(preds, targets, class_index=i)  # type: ignore
            denom = (2 * _tp + _fp + _fn).to(torch.float)
            score_cls = (2 * _tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - 1] += score_cls
        return scores


def _stat_scores(
    preds: Tensor,
    targets: Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

    if preds.ndim == targets.ndim + 1:
        preds = to_categorical(preds, argmax_dim=argmax_dim)

    tp = ((preds == class_index) * (targets == class_index)).to(torch.long).sum()
    fp = ((preds == class_index) * (targets != class_index)).to(torch.long).sum()
    tn = ((preds != class_index) * (targets != class_index)).to(torch.long).sum()
    fn = ((preds != class_index) * (targets == class_index)).to(torch.long).sum()
    sup = (targets == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


class CrossEntropy(Metric):
    """Torchmetrics cross entropy loss implementation.

    This class implements cross entropy loss as a :class:`torchmetrics.Metric` so that it can be returned by the
    :meth:`~.ComposerModel.metrics`.

    Args:
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. ``ignore_index`` is only applicable when the target
            contains class indices. Default: ``-100``.

        dist_sync_on_step (bool, optional): sync distributed metrics every step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.add_state('sum_loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total_batches', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the state with new predictions and targets."""
        # Loss calculated over samples/batch, accumulate loss over all batches
        self.sum_loss += soft_cross_entropy(preds, targets, ignore_index=self.ignore_index)
        assert isinstance(self.total_batches, Tensor)
        self.total_batches += 1

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        assert isinstance(self.total_batches, Tensor)
        assert isinstance(self.sum_loss, Tensor)
        return self.sum_loss / self.total_batches


class LossMetric(Metric):
    """Turns a torch.nn Loss Module into distributed torchmetrics Metric.

    Args:
        loss_function (callable): loss function to compute and track.

        dist_sync_on_step (bool, optional): sync distributed metrics every step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, loss_function: Callable, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.loss_function = loss_function
        self.add_state('sum_loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total_batches', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the state with new predictions and targets."""
        # Loss calculated over samples/batch, accumulate loss over all batches
        self.sum_loss += self.loss_function(preds, targets)
        self.total_batches += 1  # type: ignore

    def compute(self):
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        assert isinstance(self.total_batches, Tensor)
        assert isinstance(self.sum_loss, Tensor)
        return self.sum_loss / self.total_batches
