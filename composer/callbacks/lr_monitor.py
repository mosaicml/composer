from dataclasses import dataclass

from composer.callbacks.callback_hparams import CallbackHparams
from composer.core import Callback, Logger, State
from composer.utils import ensure_tuple


class LRMonitor(Callback):

    def batch_end(self, state: State, logger: Logger):
        assert state.optimizers is not None, "optimizers must be defined"
        for optimizer in ensure_tuple(state.optimizers):
            lrs = [group['lr'] for group in optimizer.param_groups]
            name = optimizer.__class__.__name__
            for lr in lrs:
                logger.metric_batch({f'lr-{name}': lr})


@dataclass
class LRMonitorHparams(CallbackHparams):

    def initialize_object(self) -> LRMonitor:
        return LRMonitor()
