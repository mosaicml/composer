# Copyright 2021 MosaicML. All Rights Reserved.

"""Monitor learning rate during training."""
from composer.core import Callback, Logger, State

__all__ = ["LRMonitor"]


class LRMonitor(Callback):
    """Logs the learning rate.

    This callback iterates over all optimizers and their parameter groups to log learning rate under the
    ``lr-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` key.

    Example
       >>> # constructing trainer object with this callback
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_dataloader,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ...     callbacks=[callbacks.LRMonitor()],
       ... )

    The learning rate is logged by the :class:`~composer.core.logging.logger.Logger` to the following key as described
    below.

    +---------------------------------------------+---------------------------------------+
    | Key                                         | Logged data                           |
    +=============================================+=======================================+
    |                                             | Learning rate for each optimizer and  |
    | ``lr-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` | parameter group for that optimizer is |
    |                                             | logged to a separate key              |
    +---------------------------------------------+---------------------------------------+
    """

    def __init__(self) -> None:
        super().__init__()

    def batch_end(self, state: State, logger: Logger):
        assert state.optimizers is not None, "optimizers must be defined"
        for optimizer in state.optimizers:
            lrs = [group['lr'] for group in optimizer.param_groups]
            name = optimizer.__class__.__name__
            for lr in lrs:
                for idx, lr in enumerate(lrs):
                    logger.metric_batch({f'lr-{name}/group{idx}': lr})
