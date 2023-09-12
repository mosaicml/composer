

from composer.core import Callback, State
from composer.loggers import Logger

class EvalOutputLogging(Callback):
    def __init__(self,):
        pass

    def eval_end(self, state: State, logger: Logger):
        breakpoint()