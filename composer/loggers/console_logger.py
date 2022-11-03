from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.logger import format_log_data_value, Logger
import sys
from typing import Any, Dict,Optional, TextIO, Union
from composer.utils import dist
import yaml
from composer.core import State
from composer.core.time import  TimeUnit, Time
import math

class ConsoleLogger(LoggerDestination):
    """Log metrics to the console.

    .. note::

        This logger is automatically instantiated by the trainer via the ``log_to_console``,
        and ``console_stream`` options. This logger does not need to be created manually.

    `TQDM <https://github.com/tqdm/tqdm>`_ is used to display progress bars.

    During training, the progress bar logs the batch and training loss.
    During validation, the progress bar logs the batch and validation accuracy.

    Example progress bar output::

        Epoch 1: 100%|██████████| 64/64 [00:01<00:00, 53.17it/s, loss/train=2.3023]
        Epoch 1 (val): 100%|██████████| 20/20 [00:00<00:00, 100.96it/s, accuracy/val=0.0995]

    Args:
        log_to_console (bool, optional): Whether to print logging statements to the console. (default: ``None``)
            The default behavior (when set to ``None``) only prints logging statements when ``progress_bar`` is
            ``False``.
        stream (str | TextIO, optional): The console stream to use. If a string, it can either be ``'stdout'`` or
            ``'stderr'``. (default: :attr:`sys.stderr`)
    """

    def __init__(
        self,
        log_interval: Union[int, str, Time] = '1ep',
        stream: Union[str, TextIO] = sys.stderr,
        log_traces: bool = False
    ) -> None:

        if isinstance(log_interval, int):
            log_interval = Time(log_interval, TimeUnit.EPOCH)
        if isinstance(log_interval, str):
            log_interval = Time.from_timestring(log_interval)

        if log_interval.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
            raise ValueError('The `console_log_interval` must have units of EPOCH or BATCH.')

        self.log_interval = log_interval
        # self.should_log = create_should_log_to_console_fxn(log_interval)
        # set the stream
        if isinstance(stream, str):
            if stream.lower() == 'stdout':
                stream = sys.stdout
            elif stream.lower() == 'stderr':
                stream = sys.stderr
            else:
                raise ValueError(f'stream must be one of ("stdout", "stderr", TextIO-like), got {stream}')
        
        self.should_log_traces = log_traces
        self.stream = stream
        self.state: Optional[State] = None
        self.hparams: Dict[str, Any] = {}
        self.hparams_already_logged_to_console: bool = False


    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.state = state

    def log_traces(self, traces: Dict[str, Any]):
        if self.should_log_traces:
            for trace_name, trace in traces.items():
                trace_str = format_log_data_value(trace)
                self._log_to_console(f'[trace]: {trace_name}:' + trace_str + '\n')
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        # Lazy logging of hyperparameters.
        self.hparams.update(hyperparameters)

    def _log_hparams_to_console(self):
        if dist.get_local_rank() == 0:
            self._log_to_console('*' * 30)
            self._log_to_console('Config:')
            self._log_to_console(yaml.dump(self.hparams))
            self._log_to_console('*' * 30)

    
    # def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    #     if self.should_log(self.state):
    #         for metric_name, metric_value in metrics.items():
    #             if 'metric' in metric_name or 'loss' in metric_name:
    #                 self.log_to_console(data={metric_name: metric_value})

    def epoch_end(self, state: State, logger: Logger) -> None:
        cur_epoch = int(self.state.timestamp.epoch) - 1 # epoch gets incremented right before EPOCH_END
        unit = self.log_interval.unit

        if unit == TimeUnit.EPOCH and cur_epoch % int(self.log_interval) == 0:
            if self.state.total_loss_dict:
                self.log_to_console(self.state.total_loss_dict)
            self.log_to_console(self.state.train_metric_values)


    def batch_end(self, state: State, logger: Logger) -> None:
        cur_batch = int(self.state.timestamp.batch) - 1 # batch gets incremented right before BATCH_END
        unit = self.log_interval.unit
        if unit == TimeUnit.BATCH and cur_batch % int(self.log_interval) == 0:
            if self.state.total_loss_dict:
                self.log_to_console(self.state.total_loss_dict)
            self.log_to_console(self.state.train_metric_values)

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def predict_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def eval_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def log_to_console(self, data: Dict[str, Any]) -> None:
        assert self.state is not None
        batch_in_epoch = self.state.timestamp.batch_in_epoch
        epoch = self.state.timestamp.epoch
        # cur_batch = self.state.timestamp.batch - 1
        if batch_in_epoch == 0:
            if epoch > 0:
                log_epoch = epoch - 1
                log_batch_in_epoch = int(self.state.dataloader_len) if self.state.dataloader_len is not None else batch_in_epoch
        else:
            log_batch_in_epoch = batch_in_epoch - 1
            log_epoch = epoch
        # log to console
        for data_name, data in data.items():
            data_str = format_log_data_value(data)
            if self.state.max_duration is None:
                training_progress = ''
            elif self.state.max_duration.unit == TimeUnit.EPOCH:
                if self.state.dataloader_len is None:
                    curr_progress = f'[batch={int(batch_in_epoch)}]'
                else:
                    total = int(self.state.dataloader_len)
                    curr_progress = f'[batch={int(log_batch_in_epoch)}/{total}]'

                training_progress = f'[epoch={int(log_epoch)}]{curr_progress}'
            else:
                unit = self.state.max_duration.unit
                curr_duration = int(self.state.timestamp.get(unit))
                total = self.state.max_duration.value
                training_progress = f'[{unit.name.lower()}={curr_duration}/{total}]'

            log_str = f'{training_progress}: {data_name}: {data_str}'
            self._log_to_console(log_str)

    def _log_to_console(self, log_str: str):
        """Logs to the console, avoiding interleaving with a progress bar."""
        # write directly to self.stream; no active progress bar
        print(log_str, file=self.stream, flush=True)



# def create_should_log_to_console_fxn(console_log_interval: Union[str, Time, int]):
#     if isinstance(console_log_interval, int):
#         console_log_interval = Time(console_log_interval, TimeUnit.EPOCH)
#     if isinstance(console_log_interval, str):
#         console_log_interval = Time.from_timestring(console_log_interval)

#     if console_log_interval.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
#         raise ValueError('The `console_log_interval` must have units of EPOCH or BATCH.')


#     def _should_log_to_console(state: State):
#         cur_batch = int(state.timestamp.batch)
#         cur_epoch = int(state.timestamp.epoch)
#         cur_batch_in_epoch = int(state.timestamp.batch_in_epoch)
#         unit = console_log_interval.unit
#         batches_in_an_epoch = state.dataloader_len

#         if unit == TimeUnit.EPOCH and cur_epoch % int(console_log_interval) == 0 and (cur_batch_in_epoch) == 0:
#             return True

#         if unit == TimeUnit.BATCH and cur_batch % int(console_log_interval) == 0:
#             return True

#         return False

#     return _should_log_to_console