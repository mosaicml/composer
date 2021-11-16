# Copyright 2021 MosaicML. All Rights Reserved.

from composer.core import Callback, Event, Logger, State
from composer.utils import ensure_tuple


class LRMonitor(Callback):
    """Logs the learning rate.

    Learning rates for each optimizer is logged on each batch
    under the ``lr-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` key.
    """

    def __init__(self) -> None:
        super().__init__()

    def run_event(self, event: Event, state: State, logger: Logger):
        super().run_event(event, state, logger)
        if event != Event.BATCH_END:
            return
        assert state.optimizers is not None, "optimizers must be defined"
        for optimizer in ensure_tuple(state.optimizers):
            lrs = [group['lr'] for group in optimizer.param_groups]
            name = optimizer.__class__.__name__
            for lr in lrs:
                for idx, lr in enumerate(lrs):
                    logger.metric_batch({f'lr-{name}/group{idx}': lr})
