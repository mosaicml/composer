# Copyright 2021 MosaicML. All Rights Reserved.

# type: ignore (pytorch code)
import warnings
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """Decays the learning rate of each parameter group by either a small constant or linearly increasing small warmup
    factor until the number of epoch reaches a pre-defined milestone: :attr:`warmup_iters`.

    This scheduler is adapted from PyTorch but rewritten in a non-chainable form to
    accommodate :attr:`warmup_factor=0.0`. When :attr:`last_epoch=-1`, sets initial
    lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_factor (float): The number we multiply learning rate by in the first epoch.
            If the warming up method is constant, the multiplication factor of the
            learning rate stays the same in all epochs, but, in the linear case, it
            starts increasing in the following epochs. Default: ``1./3``.
        warmup_iters (int): The number of warming up steps. Default: ``5``.
        warmup_method (str): One of ``constant`` and ``linear``. In ``constant`` mode, the
            learning rate will be multiplied with a small constant until a milestone
            defined in :attr:``warmup_iters``. In the ``linear`` case, the multiplication factor
            starts with :attr:``warmup_factor`` in the first epoch then linearly increases to
            reach 1. in the epoch number :attr:``warmup_iters``. Default: ``linear``.
        last_epoch (int): The index of the last epoch. Can be used to restore the state of the
            learning rate schedule. Default: ``-1``.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        interval (str): Frequency of ``step()`` calls, either ``step`` or ``epoch``. Default: ``epoch``.
    """

    def __init__(self,
                 optimizer,
                 warmup_factor=1.0 / 3,
                 warmup_iters=5,
                 warmup_ratio=None,
                 warmup_method="linear",
                 last_epoch=-1,
                 verbose=False,
                 interval='epoch'):
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted, but got {}".format(warmup_method))

        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_method = warmup_method
        self.interval = interval
        super(WarmUpLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Get the current learning rate for each parameter group.

        Returns:
            List of float: The current learning rate for each parameter group.
        """

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [group['lr'] * self.warmup_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.warmup_iters or
            (self.warmup_method == "constant" and self.last_epoch != self.warmup_iters)):
            return [group['lr'] for group in self.optimizer.param_groups]

        if (self.warmup_method == "constant" and self.last_epoch == self.warmup_iters):
            return [group['lr'] * (1.0 / self.warmup_factor) for group in self.optimizer.param_groups]

        # to accomodate warmup_factor = 0, use non-chainable form of lr
        return [base_lr * self.lambda_lr() for base_lr in self.base_lrs]

    def lambda_lr(self) -> float:
        """Compute the linear warmup ramp.

        Returns:
            float: Current warmup factor with which to scale the learning rate.
        """
        return (1 - self.warmup_factor) * (self.last_epoch / self.warmup_iters) + self.warmup_factor

    def _get_closed_form_lr(self) -> List[float]:
        """Get the current learning rate for each parameter group.

        Returns:
            List of float: The current learning rate for each parameter group.
        """
        return [
            base_lr * (self.warmup_factor +
                       (1 - self.warmup_factor) * min(self.warmup_iters, self.last_epoch) / self.warmup_iters *
                       (self.warmup_method == "linear") + (self.last_epoch >= self.warmup_iters) *
                       (1 - self.warmup_factor) * (self.warmup_method == "constant")) for base_lr in self.base_lrs
        ]

    def scale_schedule(self, ssr: float) -> None:
        pass


class LinearLR(_LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1, verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [
            group['lr'] * (1. + (self.end_factor - self.start_factor) /
                           (self.total_iters * self.start_factor + (self.last_epoch - 1) *
                            (self.end_factor - self.start_factor))) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr *
            (self.start_factor +
             (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
            for base_lr in self.base_lrs
        ]
