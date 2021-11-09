# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import MagicMock
from composer.datasets.synthetic import SyntheticDataLabelType

import pytest

from composer.benchmarker.benchmarker import Benchmarker
from tests.fixtures.models import SimpleBatchPairModel


@pytest.mark.timeout(90)
def test_benchmarking(dummy_model: SimpleBatchPairModel):
    log_destination = MagicMock()
    log_destination.will_log.return_value = True

    benchmarker = Benchmarker(model=dummy_model,
                              data_shape=dummy_model.in_shape,
                              total_batch_size=16,
                              grad_accum=1,
                              label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
                              num_classes=dummy_model.num_classes,
                              log_destinations=[log_destination])

    benchmarker.run_timing_benchmark()

    num_step_called = 0
    num_wall_clock_train_called = 0
    for log_call in log_destination.log_metric.mock_calls:
        metrics = log_call[1][3]
        if "wall_clock_train" in metrics:
            num_wall_clock_train_called += 1
        if "throughput/step" in metrics:
            num_step_called += 1

    assert num_step_called == 4
    assert num_wall_clock_train_called == 2
