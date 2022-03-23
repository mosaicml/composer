import os
from composer import Trainer
from composer.callbacks import MLPerfCallback
from tests.common import SimpleModel, RandomClassificationDataset
import pytest
from torch.utils.data import DataLoader


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
        'seed': 0,
        'deterministic_mode': True,  # testing equivalence
        'loggers': [],  # no progress bar
        'callbacks': []
    }


@pytest.mark.filterwarnings(
    "ignore: DeprecationWarning",)
def test_mlperf_callback(config, tmpdir):
    pytest.importorskip("mlperf_logging")
    result_folder = os.path.join(tmpdir, "results")
    os.mkdir(result_folder)

    for run in range(5):
        filename = os.path.join(result_folder, f"result_{run}.txt")
        mlperf_callback = MLPerfCallback(filename=filename)
        config['callbacks'].append(mlperf_callback)

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