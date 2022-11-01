from composer.loggers.logger_destination import LoggerDestination
import sys
from typing import Any, Dict, List, Optional, TextIO, Union


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
        log_to_console: Optional[bool] = None,
        stream: Union[str, TextIO] = sys.stderr,
    ) -> None:

        self.should_log_to_console = log_to_console
        if self.should_log_to_console is None:
            self.should_log_to_console = not progress_bar

        # set the stream
        if isinstance(stream, str):
            if stream.lower() == 'stdout':
                stream = sys.stdout
            elif stream.lower() == 'stderr':
                stream = sys.stderr
            else:
                raise ValueError(f'stream must be one of ("stdout", "stderr", TextIO-like), got {stream}')
        
        self.stream = stream
        self.state: Optional[State] = None
        self.hparams: Dict[str, Any] = {}
        self.hparams_already_logged_to_console: bool = False


    def log_traces(self, traces: Dict[str, Any]):
        if self.should_log_to_console:
            for trace_name, trace in traces.items():
                trace_str = format_log_data_value(trace)
                self._log_to_console(f'[trace]: {trace_name}:' + trace_str + '\n')
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        # Lazy logging of hyperparameters.
        self.hparams.update(hyperparameters)

    def _log_hparams_to_console(self):
        if self.should_log_to_console or self._show_pbar:
            if dist.get_local_rank() == 0:
                self._log_to_console('*' * 30)
                self._log_to_console('Config:')
                self._log_to_console(yaml.dump(self.hparams))
                self._log_to_console('*' * 30)

    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for metric_name, metric_value in metrics.items():
            # Only log metrics and losses to pbar.
            if 'metric' in metric_name or 'loss' in metric_name:
                if self._show_pbar:
                    self.log_to_pbar(data={metric_name: metric_value})
            if self.should_log_to_console:
                self.log_to_console(data={metric_name: metric_value})