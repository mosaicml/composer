# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
from typing import List

import pytest

from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.synthetic import SyntheticBatchPairDataset, SyntheticDataLabelType


@pytest.mark.vision
@pytest.mark.timeout(30)
def test_write_ffcv_dataset(tmpdir: pathlib.Path, dummy_in_shape: List[int]):
    dataset = SyntheticBatchPairDataset(total_dataset_size=1,
                                        num_classes=1,
                                        data_shape=list(dummy_in_shape),
                                        label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
                                        num_unique_samples_to_create=1)
    output_file = str(tmpdir / "ffcv")
    write_ffcv_dataset(dataset, write_path=output_file, num_workers=1)
    assert os.path.exists(output_file)
