from composer import Callback
from composer.core import State
from composer.core.time import Time, TimeUnit
from composer.loggers import Logger


class ThresholdStopper(Callback):

    def __init__(self,):
        self.wait = 0

    def epoch_end(self, state: State, logger: Logger) -> None:
        if state.timer >= Time(2, TimeUnit.EPOCH):
            state.max_duration = state.timer.batch
