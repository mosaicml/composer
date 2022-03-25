import logging

import numpy as np
import pytest
from torch.utils.data import DataLoader

from composer import Trainer
from composer.callbacks import MLPerfCallback
from tests.common import RandomClassificationDataset, SimpleModel

logging.basicConfig(filename="/Users/hanlintang/composer/package_checker.log", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
formatter = logging.Formatter("%(levelname)s - %(message)s")
logging.getLogger().handlers[0].setFormatter(formatter)
logging.getLogger().handlers[1].setFormatter(formatter)


@pytest.fixture
def config():
    """Returns the reference config."""

    return {
        'model': SimpleModel(),
        'train_dataloader': DataLoader(
            dataset=RandomClassificationDataset(),
            batch_size=4,
            shuffle=False,
        ),
        'eval_dataloader': DataLoader(
            dataset=RandomClassificationDataset(),
            shuffle=False,
        ),
        'max_duration': '2ep',
        'deterministic_mode': True,  # testing equivalence
        'loggers': [],  # no progress bar
        'callbacks': []
    }


@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",)
def test_mlperf_callback(config, tmpdir):
    tmpdir = 'mlperf_results'
    pytest.importorskip("mlperf_logging")

    for run in range(5):
        mlperf_callback = MLPerfCallback(root_folder=tmpdir, num_result=run)
        config['callbacks'] = [mlperf_callback]
        config['seed'] = np.random.randint(2e5)  # mlperf seeds are released near submission deadline
        trainer = Trainer(**config)
        trainer.fit()

    # run result checker
    from mlperf_logging.package_checker.package_checker import check_training_package

    check_training_package(
        folder=tmpdir,
        usage="training",
        ruleset="1.1.0",
        werror=True,
        quiet=False,
        rcp_bypass=False,
        rcp_bert_train_samples=False,
        log_output="package_checker.log",
    )