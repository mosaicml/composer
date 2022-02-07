from torch.optim.lr_scheduler import LambdaLR

from composer.composer.utils.iter_helpers import ensure_tuple
from composer.core import State
from composer.core.types import ComposerSchedulerFn, Optimizers, Scheduler

COMPOSER_SCHEDULER_FLAG = '__composer_scheduler'


def compile_scheduler(scheduler: ComposerSchedulerFn, optimizers: Optimizers, state: State) -> Scheduler:

    optimizers = ensure_tuple(optimizers)
    if len(optimizers) != 1:
        raise NotImplementedError("Providing ComposerSchedulerFn requires exactly one optimizer.")
    optimizer = optimizers[0]

    def scheduler_fn(epoch: int) -> float:
        del epoch  # unused
        return scheduler(state)

    lambda_scheduler = LambdaLR(optimizer, scheduler_fn)
    setattr(lambda_scheduler, COMPOSER_SCHEDULER_FLAG, True)

    return lambda_scheduler


def should_step_scheduler(scheduler: Scheduler, is_epoch: bool):

    return is_epoch or getattr(scheduler, COMPOSER_SCHEDULER_FLAG, True)
