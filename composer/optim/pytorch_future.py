# Copyright 2021 MosaicML. All Rights Reserved.

# type: ignore (pytorch code)
import warnings
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """Decays the learning rate of each parameter group by either a small constant
    or linearly increasing small warmup factor until the number of epoch reaches a
    pre-defined milestone: :attr:`warmup_iters`.

    This scheduler is adapted from PyTorch but rewritten in a non-chainable form to
    accomodate :attr:`warmup_factor=0.0`. When :attr:`last_epoch=-1`, sets initial
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
        interval (str): Frequency of ``step()`` calls, either ``step`` or ``epoch``. Default: ``step``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = WarmUpLR(self.opt, warmup_factor=0.5, warmup_iters=4, warmup_method="linear")
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.025    if epoch == 1
        >>> # lr = 0.025    if epoch == 2
        >>> # lr = 0.025    if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = WarmUpLR(self.opt, warmup_factor=0.5, warmup_iters=4, warmup_method="constant")
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self,
                 optimizer,
                 warmup_factor=1.0 / 3,
                 warmup_iters=5,
                 warmup_method="linear",
                 last_epoch=-1,
                 verbose=False,
                 interval='step'):
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted, but " "got {}".format(warmup_method))
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.interval = interval
        super(WarmUpLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """ Get the current learning rate for each parameter group.

        Returns:
            List of float: The current learning rate for each parameter group.
        """

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

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
        """ Compute the linear warmup ramp.

        Returns:
            float: Current warmup factor with which to scale the learning rate.
        """
        return (1 - self.warmup_factor) * (self.last_epoch / self.warmup_iters) + self.warmup_factor

    def _get_closed_form_lr(self) -> List[float]:
        """ Get the current learning rate for each parameter group.

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
