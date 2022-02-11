from typing import Callable

from torch.optim.lr_scheduler import LambdaLR

from composer.core import State
from composer.core.types import Optimizers, Scheduler
from composer.utils.iter_helpers import ensure_tuple

ComposerSchedulerFn = Callable[[State], float]


def compile_scheduler(scheduler: ComposerSchedulerFn, optimizers: Optimizers, state: State) -> Scheduler:

    optimizers = ensure_tuple(optimizers)
    if len(optimizers) != 1:
        raise NotImplementedError("Providing ComposerSchedulerFn requires exactly one optimizer.")
    optimizer = optimizers[0]

    def scheduler_fn(epoch: int) -> float:
        del epoch  # unused
        return scheduler(state)

    lambda_scheduler = LambdaLR(optimizer, scheduler_fn)

    return lambda_scheduler
