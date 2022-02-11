# Copyright 2021 MosaicML. All Rights Reserved.

"""Monitor learning rate during training."""
from composer.core import Callback, Logger, State

__all__ = ["LRMonitor"]


class LRMonitor(Callback):
    """Logs the learning rate to a key.

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
        """Called on the :attr:`~composer.core.event.Event.BATCH_END` event.

        Walks through all optimizers and their parameter groups to log learning rate under the
        ``lr-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` key.

        Args:
            state (State): The :class:`~composer.core.state.State` object
                used during training.
            logger (Logger):
                The :class:`~composer.core.logging.logger.Logger` object.
        """
        assert state.optimizers is not None, "optimizers must be defined"
        for optimizer in state.optimizers:
            lrs = [group['lr'] for group in optimizer.param_groups]
            name = optimizer.__class__.__name__
            for lr in lrs:
                for idx, lr in enumerate(lrs):
                    logger.metric_batch({f'lr-{name}/group{idx}': lr})
