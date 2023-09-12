

from composer.core import Callback, State
from composer.loggers import Logger

class EvalOutputLogging(Callback):
    def __init__(self,):
        pass

    def prep_response_cache(self, state, cache):
        benchmark = state.dataloader_label
        for metric in state.eval_metrics[benchmark].values():
            if hasattr(metric, 'set_response_cache'):
                metric.set_response_cache(cache)
                
    def eval_start(self, state: State, logger: Logger) -> None:
        self.prep_response_cache(state, True)
    
    def eval_end(self, state: State, logger: Logger) -> None:
        if hasattr(state.dataloader, 'dataset') and hasattr(state.dataloader.dataset, 'tokenizer'):
            tokenizer = state.dataloader.dataset.tokenizer
            benchmark = state.dataloader_label
            for _,metric in state.eval_metrics[benchmark].items():
                if hasattr(metric, 'format_response_cache'):
                    columns, rows = metric.format_response_cache(tokenizer)
                    if columns is not None and rows is not None:
                        logger.log_table(
                            columns=columns, rows=rows, name=f"icl_outputs/{benchmark}"
                        )
        self.prep_response_cache(state, False)