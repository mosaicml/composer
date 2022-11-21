
from tests.common import RandomClassificationDataset, SimpleModel
from pathlib import Path
from torch.utils.data import DataLoader
from composer.trainer import Trainer
import pytest
import re
from composer.utils import ensure_tuple
import math

@pytest.fixture
def console_logger_test_file_path(tmp_path) -> str:
    return str(Path(tmp_path) / Path('console_test'))

@pytest.fixture
def console_logger_test_stream(console_logger_test_file_path):
    return open(console_logger_test_file_path, 'w')


@pytest.mark.parametrize('log_interval_unit', ['ba', 'ep'])
@pytest.mark.parametrize('max_duration_unit', ['ba', 'ep'])
@pytest.mark.parametrize('log_interval', [1, 2, 3])
@pytest.mark.parametrize('max_duration', [8, 9, 10, 11])
def test_console_logger_interval(console_logger_test_stream,
                                console_logger_test_file_path,
                                log_interval,
                                max_duration,
                                log_interval_unit,
                                max_duration_unit):


    batch_size = 4
    dataset_size = 17
    batches_per_epoch = math.ceil(dataset_size / batch_size)

    model=SimpleModel()
    # model.train_metrics = 
    trainer = Trainer(model=model,
                    console_stream=console_logger_test_stream,
                    console_log_interval=f'{log_interval}{log_interval_unit}',
                    log_to_console=True,
                    progress_bar=False,
                    train_dataloader=DataLoader(RandomClassificationDataset(size=dataset_size),
                                                                            batch_size=batch_size),                                                  
                    max_duration=f'{max_duration}{max_duration_unit}')
    trainer.fit()
    console_logger_test_stream.flush()
    console_logger_test_stream.close()
    
    with open(console_logger_test_file_path, 'r') as f:
        lines = f.readlines()

    # Make a regular expression for matches for any line that contains "Train" followed by
    # a colon.
    reg_exp = re.compile('Train *:*')
    actual_num_log_lines = sum([1 if bool(reg_exp.search(line)) else 0 for line in lines])

    num_metrics = len(ensure_tuple(model.train_metrics))
    num_losses = 1
    num_metrics_and_losses_per_logging_event = num_metrics + num_losses

    if log_interval_unit == max_duration_unit:
        expected_num_logging_events = max_duration // log_interval
    elif log_interval_unit == 'ba' and max_duration_unit == 'ep':
        expected_num_logging_events = (batches_per_epoch * max_duration) // log_interval
    else: # for the case where log_interval_unit == 'ep' and max_duration == 'ba'.
        total_epochs = max_duration // batches_per_epoch
        expected_num_logging_events = total_epochs // log_interval


    expected_num_lines = expected_num_logging_events * num_metrics_and_losses_per_logging_event

    assert actual_num_log_lines == expected_num_lines
    
