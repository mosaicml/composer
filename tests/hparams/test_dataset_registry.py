# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import pytest

from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.dataset_hparams_registry import (ADE20kDatasetHparams, BratsDatasetHparams, C4DatasetHparams,
                                                        CIFAR10DatasetHparams, COCODatasetHparams, GLUEHparams,
                                                        ImagenetDatasetHparams, LMDatasetHparams, MNISTDatasetHparams,
                                                        StreamingADE20kHparams, StreamingC4Hparams,
                                                        StreamingCIFAR10Hparams, StreamingCOCOHparams,
                                                        StreamingImageNet1kHparams, dataset_registry)
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin

# for testing, we provide values for required hparams fields
# to initialize test hparams objects
default_required_fields: Dict[Type[DatasetHparams], Callable[[], DatasetHparams]] = {
    #  hparams with empty dicts have no required fields
    CIFAR10DatasetHparams:
        lambda: CIFAR10DatasetHparams(
            is_train=False,
            download=False,
        ),
    StreamingCIFAR10Hparams:
        lambda: StreamingCIFAR10Hparams(split='val'),
    ADE20kDatasetHparams:
        lambda: ADE20kDatasetHparams(split='val'),
    StreamingADE20kHparams:
        lambda: StreamingADE20kHparams(split='val'),
    BratsDatasetHparams:
        lambda: BratsDatasetHparams(is_train=False,),
    ImagenetDatasetHparams:
        lambda: ImagenetDatasetHparams(
            is_train=False,
            resize_size=256,
            crop_size=224,
        ),
    StreamingImageNet1kHparams:
        lambda: StreamingImageNet1kHparams(split='val', resize_size=256, crop_size=224),
    MNISTDatasetHparams:
        lambda: MNISTDatasetHparams(
            is_train=False,
            download=False,
        ),
    LMDatasetHparams:
        lambda: LMDatasetHparams(
            datadir=['hello'],  # type: ignore # need to remove the datadir from the base class.
            split='train',
            tokenizer_name='gpt2',
            use_masked_lm=False,
        ),
    GLUEHparams:
        lambda: GLUEHparams(
            task='rte',
            tokenizer_name='bert-base-uncased',
            split='train',
        ),
    COCODatasetHparams:
        lambda: COCODatasetHparams(
            is_train=False,
            datadir='hello',
            drop_last=False,
            shuffle=False,
        ),
    StreamingCOCOHparams:
        lambda: StreamingCOCOHparams(split='val'),
    C4DatasetHparams:
        lambda: C4DatasetHparams(
            split='train',
            num_samples=1000,
            max_seq_len=100,
            tokenizer_name='gpt2',
            group_method='concat',
        ),
    StreamingC4Hparams:
        lambda: StreamingC4Hparams(split='val'),
}


@pytest.mark.parametrize('dataset_name', dataset_registry.keys())
def test_dataset(dataset_name: str, dummy_dataloader_hparams: DataLoaderHparams) -> None:
    hparams_cls = dataset_registry[dataset_name]
    hparams = default_required_fields[hparams_cls]()
    if not isinstance(hparams, SyntheticHparamsMixin):
        pytest.xfail(f'{hparams.__class__.__name__} does not support synthetic data')
    if isinstance(hparams, GLUEHparams) or isinstance(hparams, LMDatasetHparams):
        pytest.importorskip('transformers')
        pytest.importorskip('datasets')
        pytest.importorskip('tokenizers')

    assert isinstance(hparams, SyntheticHparamsMixin)

    hparams.use_synthetic = True

    hparams.initialize_object(batch_size=1, dataloader_hparams=dummy_dataloader_hparams)
