# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest

from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.synthetic import SyntheticDataLabelType, SyntheticPILDataset


@pytest.mark.vision
def test_write_ffcv_dataset(tmp_path: pathlib.Path):
    dataset = SyntheticPILDataset(total_dataset_size=1,
                                  num_classes=1,
                                  data_shape=[1, 1, 3],
                                  label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
                                  num_unique_samples_to_create=1)
    output_file = str(tmp_path / 'ffcv')
    write_ffcv_dataset(dataset, write_path=output_file, num_workers=1)
    assert os.path.exists(output_file)
