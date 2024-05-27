from megatron.core.utils import StragglerDetector
from composer.core import Callback, State, Event, Time
from composer.utils import dist
from typing import List, Union
from dataclasses import dataclass
import time
from composer.models.base import ComposerModel
import os

__all__ = ["globalStragglerDetector"]


class globalStragglerDetector(Callback):

    def __init__(self) -> None:
        self.stimer = None
        self.total_flops = 0.0
        self.log_interval = 0
        self.start_time = None

    def init(self, state: State, logger: Logger) -> None:
        self.stimer = StragglerDetector()
        port = int(os.environ.get('MASTER_PORT'))
        rank = dist.get_global_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            self.stimer.configure(world_size, rank, enabled=True, port=port)
        else:
            self.stimer.configure(world_size, rank, enabled=True)
        

    def before_train_batch(self, state: State, logger: Logger):
        self.start_time = time.time()

    def after_train_batch(self, state: State, logger: Logger):
        # Calculate duration of the current batch
        batch_time = (time.time() - self.start_time) * 1000
        self.log_interval = int(batch_time)

        # Compute flops stats if model has flops_per_batch
        composer_model = state.model
        if not isinstance(composer_model, ComposerModel):
            composer_model = composer_model.module
        if hasattr(composer_model, 'flops_per_batch'):
            model_flops_per_batch = composer_model.flops_per_batch  # type: ignore
            if not isinstance(model_flops_per_batch, Callable):
                raise TypeError(
                    'flops_per_batch must a callable accepting a batch and '
                    f'returning an int or float. Instead, got {type(model_flops_per_batch)}.',
                )
            device_flops_per_batch = model_flops_per_batch(state.batch)
            self.stimer.report(total_flops=device_flops_per_batch, log_interval=self.log_interval)
            self.total_flops = 0.0

        else:
            raise ValueError("The 'flops_per_batch' attribute is not present in this model; StragglerDetector requires tracking flops per batch.")

        









