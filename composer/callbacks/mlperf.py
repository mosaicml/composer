from typing import Dict
from composer import Callback, State
from composer.loggers import Logger
try:
    from mlperf_logging import mllog
    from mlperf_logging.mllog import constants
    mlperf_available = True
except ImportError:
    mlperf_available = False

from mlperf_logging import mllog
from composer.utils import dist

BENCHMARKS = ("resnet")
DIVISIONS = ("open")
STATUS = ("onprem", "cloud", "preview")


def rank_zero() -> bool:
    return dist.get_global_rank() == 0


class MLPerfCallback(Callback):

    def __init__(
        self,
        filename: str,
        org: str = "MosaicML",
        platform: str = "A100",
        benchmark: str = "resnet",
        division: str = "open",
        submission_status: str = "onprem",
        target: float = 75.9,
    ) -> None:

        if benchmark not in BENCHMARKS:
            raise ValueError(f"benchmark: {benchmark} must be one of {BENCHMARKS}")
        if division not in DIVISIONS:
            raise ValueError(f"division: {division} must be one of {DIVISIONS}")
        if submission_status not in STATUS:
            raise ValueError(f"status: {submission_status} must be one of {STATUS}")
        if not mlperf_available:
            raise ValueError("MLperf logger is required")
        self.mllogger = mllog.get_mllogger()
        self.target = target

        mllog.config(filename=filename)

        # TODO: implement cache clearing
        self.mllogger.start(key=mllog.constants.CACHE_CLEAR)
        self.mllogger.start(key=mllog.constants.INIT_START)

        if rank_zero():
            self._log_dict({
                constants.SUBMISSION_BENCHMARK: benchmark,
                constants.SUBMISSION_DIVISION: division,
                constants.SUBMISSION_ORG: org,
                constants.SUBMISSION_PLATFORM: platform,
                constants.SUBMISSION_STATUS: submission_status,
            })

    def _log_dict(self, data: Dict):
        for key, value in data.items():
            self.mllogger.event(key=key, value=value)

    def fit_start(self, state: State, logger: Logger) -> None:
        if rank_zero():
            if state.train_dataloader.batch_size is None:
                raise ValueError("Batch size is required to be set for dataloader.")

            self._log_dict({
                constants.SEED: state.seed,
                constants.GLOBAL_BATCH_SIZE: state.train_dataloader.batch_size * dist.get_world_size(),
                constants.GRADIENT_ACCUMULATION_STEPS: state.grad_accum,
            })

        self.mllogger.event(key=constants.INIT_STOP)

        dist.barrier()
        if rank_zero():
            self.mllogger.event(key=constants.RUN_START)

    def epoch_start(self, state: State, logger: Logger) -> None:
        if rank_zero():
            self.mllogger.event(key=constants.EPOCH_START, metadata={'epoch_num': state.timer.epoch.value})
            self.mllogger.event(key=constants.BLOCK_START,
                                metadata={
                                    'first_epoch_num': state.timer.epoch.value,
                                    'epoch_count': 1
                                })

    def epoch_end(self, state: State, logger: Logger) -> None:
        if rank_zero():
            self.mllogger.event(key=constants.EPOCH_STOP, metadata={'epoch_num': state.timer.epoch.value})

    def eval_start(self, state: State, logger: Logger) -> None:
        if rank_zero():
            self.mllogger.event(key=constants.EVAL_START, metadata={'epoch_num': state.timer.epoch.value})

    def eval_end(self, state: State, logger: Logger) -> None:
        if rank_zero():
            self.mllogger.event(key=constants.EVAL_STOP, metadata={'epoch_num': state.timer.epoch.value})
            self.mllogger.event(key=constants.EVAL_ACCURACY, value=0.99)
            self.mllogger.event(key=constants.BLOCK_STOP, metadata={'first_epoch_num': state.timer.epoch.value})

        accuracy = 0.99
        if accuracy > self.target:
            self.mllogger.event(key=constants.RUN_STOP, metadata={"status": "success"})
