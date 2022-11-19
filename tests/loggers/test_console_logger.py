from composer.loggers import ConsoleLogger
from tests.common import RandomClassificationDataset, SimpleModel
from pathlib import Path
from torch.utils.data import DataLoader
from composer.trainer import Trainer
import pytest 

@pytest.fixture
def console_logger_test_file_path(tmp_path) -> str:
    return str(Path(tmp_path) / Path('console_test'))

@pytest.fixture
def console_logger_test_stream(console_logger_test_file_path):
    return open(console_logger_test_file_path, 'w')

@pytest.mark.parametrize('log_interval_by_batch', [1, 2, 3])
@pytest.mark.parametrize('max_duration_in_batches', [8, 9, 10, 11])
def test_console_logger_interval(console_logger_test_stream,
                                console_logger_test_file_path,
                                log_interval_by_batch,
                                max_duration_in_batches):


    trainer = Trainer(model=SimpleModel(),
                    console_stream=console_logger_test_stream,
                    console_log_interval=f'{log_interval_by_batch}ba',
                    log_to_console=True,
                    progress_bar=False,
                    train_dataloader=DataLoader(RandomClassificationDataset()),
                    max_duration=f'{max_duration_in_batches }ba')
    trainer.fit()
    console_logger_test_stream.flush()
    console_logger_test_stream.close()
    
    with open(console_logger_test_file_path, 'r') as f:
        lines = f.readlines()

    num_log_lines = sum([1 for line in lines if "Train Accuracy" in line])

    assert num_log_lines == max_duration_in_batches // log_interval_by_batch
    
