# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch

from composer.datasets.synthetic import (SyntheticBatchPairDataset, SyntheticDataLabelType, SyntheticDataType,
                                         SyntheticPILDataset)


@pytest.mark.parametrize('data_type', [
    SyntheticDataType.GAUSSIAN,
    SyntheticDataType.SEPARABLE,
])
@pytest.mark.parametrize('label_type', [
    SyntheticDataLabelType.CLASSIFICATION_ONE_HOT,
    SyntheticDataLabelType.CLASSIFICATION_INT,
])
def test_synthetic_batch_pair_creation(data_type: SyntheticDataType, label_type: SyntheticDataLabelType):
    if data_type == SyntheticDataType.SEPARABLE:
        if label_type != SyntheticDataLabelType.CLASSIFICATION_INT:
            pytest.skip('Separable data requires classification int labels')
        num_classes = 2
        label_shape = None
    else:
        num_classes = 10
        label_shape = (1, 10, 12)

    if data_type == SyntheticDataType.GAUSSIAN and label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
        pytest.xfail('classification_int is not currently supported with gaussian data')

    dataset_size = 1000
    data_shape = (3, 32, 32)
    num_samples_to_create = 10
    dataset = SyntheticBatchPairDataset(total_dataset_size=dataset_size,
                                        data_shape=data_shape,
                                        num_unique_samples_to_create=num_samples_to_create,
                                        data_type=data_type,
                                        label_type=label_type,
                                        num_classes=num_classes,
                                        label_shape=label_shape)
    assert len(dataset) == dataset_size

    # verify datapoints are correct
    x, y = dataset[0]
    assert x.size() == data_shape
    if label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
        assert isinstance(y.item(), int)
    elif label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
        assert y.size() == (num_classes,)
        assert torch.min(y) == 0
        assert torch.max(y) == 1

    # check that points were allocated in memory after the first call to __getitem__
    assert dataset.input_data is not None
    assert dataset.input_target is not None
    # check that the correct number of points were allocated in memory
    assert dataset.input_data.size()[0] == num_samples_to_create
    assert dataset.input_target.size()[0] == num_samples_to_create

    # verify that you can getch points outside the num_samples_to_create range
    # (still within the total dataset size range)
    x, y = dataset[num_samples_to_create + 1]
    assert x is not None
    assert y is not None


@pytest.mark.parametrize('label_type', [
    SyntheticDataLabelType.CLASSIFICATION_ONE_HOT,
    SyntheticDataLabelType.CLASSIFICATION_INT,
])
@pytest.mark.parametrize('num_classes', [None, 0])
def test_synthetic_classification_param_validation(label_type: SyntheticDataLabelType, num_classes: Optional[int]):
    with pytest.raises(ValueError):
        SyntheticBatchPairDataset(total_dataset_size=10,
                                  data_shape=(2, 2),
                                  label_type=label_type,
                                  num_classes=num_classes)


@pytest.mark.parametrize('data_type', [
    SyntheticDataType.GAUSSIAN,
    SyntheticDataType.SEPARABLE,
])
@pytest.mark.parametrize('label_type', [
    SyntheticDataLabelType.CLASSIFICATION_ONE_HOT,
    SyntheticDataLabelType.CLASSIFICATION_INT,
])
def test_synthetic_image_data_creation(data_type: SyntheticDataType, label_type: SyntheticDataLabelType):
    if data_type == SyntheticDataType.SEPARABLE:
        if label_type != SyntheticDataLabelType.CLASSIFICATION_INT:
            pytest.skip('Seperable data requires classification int labels')
        num_classes = 2
        label_shape = None
    else:
        num_classes = 10
        label_shape = (1, 10, 12)

    if data_type == SyntheticDataType.GAUSSIAN and label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
        pytest.xfail('classification_int is not currently supported with gaussian data')

    dataset_size = 1000
    data_shape = (32, 32)
    num_samples_to_create = 100
    dataset = SyntheticPILDataset(total_dataset_size=dataset_size,
                                  data_shape=data_shape,
                                  num_unique_samples_to_create=num_samples_to_create,
                                  data_type=data_type,
                                  label_type=label_type,
                                  num_classes=num_classes,
                                  label_shape=label_shape)
    assert len(dataset) == dataset_size

    # verify datapoints are correct
    x, y = dataset[0]
    assert x.size == data_shape
    if label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
        assert isinstance(y.item(), int)
    elif label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
        assert y.size() == (num_classes,)
        assert torch.min(y) == 0
        assert torch.max(y) == 1

    # check that points were allocated in memory after the first call to __getitem__
    assert dataset._dataset.input_data is not None
    assert dataset._dataset.input_target is not None
    # check that the correct number of points were allocated in memory
    assert dataset._dataset.input_data.shape[0] == num_samples_to_create
    assert dataset._dataset.input_target.shape[0] == num_samples_to_create

    # verify that you can getch points outside the num_samples_to_create range
    # (still within the total dataset size range)
    x, y = dataset[num_samples_to_create + 1]
    assert x is not None
    assert y is not None
