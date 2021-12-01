# Copyright 2021 MosaicML. All Rights Reserved.

import pytest

from composer.datasets.hparams import DatasetHparams


def test_common_fields_raise_if_not_defined(dummy_train_dataset_hparams: DatasetHparams):
    with pytest.raises(AttributeError):
        # dummy_train_dataset_hparams does not support datadir
        dummy_train_dataset_hparams.datadir = ""


def test_common_fields_work_if_defined(dummy_train_dataset_hparams: DatasetHparams):
    # dummy_train_dataset_hparams supports shuffle
    dummy_train_dataset_hparams.shuffle = False
    assert not dummy_train_dataset_hparams.shuffle
