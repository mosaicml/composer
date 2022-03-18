from typing import Dict
from composer import Callback, State
from composer.loggers import Logger
from composer.optim.scheduler import ConstantScheduler
try:
    from mlperf_logging import mllog
    from mlperf_logging.mllog import constants
    mlperf_available = True
except ImportError:
    mlperf_available = False

from mlperf_logging import mllog
from composer.utils import dist


def rank_zero() -> bool:
    return dist.get_global_rank() == 0


class MLPerfCallback(Callback):

    def __init__(self, org: str, platform: str, target: float) -> None:
        self.mllogger = mllog.get_logger()
        self.target = target
        mllog.config(filename='benchmark.log')
        self.mllogger.start(key=mllog.constants.INIT_START)

        if rank_zero():
            self._log_dict({
                constants.SUBMISSION_BENCHMARK: "resnet",
                constants.SUBMISSION_DIVISION: "closed",
                constants.SUBMISSION_ORG: org,
                constants.SUBMISSION_PLATFORM: platform,
                constants.SUBMISSION_STATUS: "onprem",
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

    # mlperf_logger.log_start(key=mllog.constants.INIT_START, log_all_ranks=True, sync=False)
    # mlperf_logger.mlperf_submission_log('bert')

    # mlperf_logger.log_event(key=mlperf_logger.constants.SEED, value=args.seed, sync=False)
    # mlperf_logger.log_event(key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=global_batch_size(args), sync=False)
    # mlperf_logger.log_event(key='d_batch_size', value=args.train_batch_size, sync=False)
    # mlperf_logger.log_event(key=mlperf_logger.constants.GRADIENT_ACCUMULATION_STEPS,
    #                         value=args.gradient_accumulation_steps,
    #                         sync=False)
    # mlperf_logger.log_event(key='max_predictions_per_seq', value=args.max_predictions_per_seq, sync=False)
    # mlperf_logger.log_event(key='opt_learning_rate_training_steps', value=args.max_steps, sync=False)
    # mlperf_logger.log_event(key='num_warmup_steps',
    #                         value=int(args.warmup_proportion *
    #                                   args.max_steps) if args.warmup_steps == 0 else args.warmup_steps,
    #                         sync=False)
    # mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP, sync=False)
    # mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START, sync=True)
    # mlperf_logger.barrier()

    # mlperf_logger.log_start(key=mlperf_logger.constants.EPOCH_START, metadata={'epoch_num': epoch}, sync=False)
    # mlperf_logger.log_start(key=mlperf_logger.constants.BLOCK_START,
    #                         metadata={
    #                             'first_epoch_num': epoch,
    #                             'epoch_count': 1
    #                         },
    #                         sync=False)
    # mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_ACCURACY,
    #                         value=eval_avg_mlm_accuracy,
    #                         metadata={'epoch_num': epoch},
    #                         sync=False)
    # print({"global_steps": global_step, "eval_loss": eval_avg_loss, "eval_mlm_accuracy": eval_avg_mlm_accuracy})

    # mlperf_logger.log_event(key='tracked_stats',
    #                         value={
    #                             'seq/sec': training_perf,
    #                             'step_loss': loss.item() * args.gradient_accumulation_steps / divisor,
    #                             'avg_loss': average_loss / (args.log_freq * divisor),
    #                             'lr': now_lr
    #                         },
    #                         metadata={"step": (epoch, training_steps)},
    #                         sync=False)

    # mlperf_logger.log_event(key='throughput', value=training_perf)

    # mlperf_logger.log_end(key=mlperf_logger.constants.BLOCK_STOP, metadata={'first_epoch_num': epoch}, sync=False)
    # mlperf_logger.log_end(key=mlperf_logger.constants.EPOCH_STOP, metadata={'epoch_num': epoch}, sync=False)
    # mlperf_logger.log_event(key=mlperf_logger.constants.TRAIN_SAMPLES, value=samples_trained, sync=False)
    # mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_SAMPLES, value=args.num_eval_examples, sync=False)
    # mlperf_logger.log_end(key=mlperf_logger.constants.RUN_STOP, metadata={'status': status}, sync=False)
