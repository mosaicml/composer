import time

from composer.utils import dist
from composer.loggers import MLFlowLogger


dist.initialize_dist('gpu', timeout=10)

run_name = 'testrun1'

mlflow_logger = MLFlowLogger(
    experiment_name='test_experiment',
    run_name = run_name,
)

class State():
    def __init__(self):
        self.run_name = run_name

mlflow_logger.init(State(), None)

if dist.get_global_rank() == 1:
    time.sleep(300)

dist.barrier()

print('Rank', dist.get_global_rank(), 'done') 