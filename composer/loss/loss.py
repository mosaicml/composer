from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.nn import functional as F

__all__ = ["soft_cross_entropy"]


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


def ensure_targets_one_hot(input: Tensor, targets: Tensor) -> Tensor:
    if _infer_target_type(input, targets) == 'indices':
        targets = F.one_hot(targets, num_classes=input.shape[1])
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
        assert ignore_index == -100, "ignore_index not supported."
        probs = -1 * (target * F.log_softmax(input, dim=1))

        if weight is not None:
            probs *= weight / weight.sum()  # allow broadcast along batch dim

        probs = probs.sum(dim=1)

        if reduction == 'sum':
            probs = probs.sum(dim=0)
        elif reduction == 'mean':
            probs = probs.mean(dim=0)

        return probs
    else:
        raise ValueError(f"Unrecognized target type {target_type}")