# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import time
from typing import Any, Dict, Optional, Tuple

import pytest
import pytest_httpserver
from torch.utils.data import DataLoader

from composer.datasets.ade20k import StreamingADE20k
from composer.datasets.c4 import StreamingC4
from composer.datasets.cifar import StreamingCIFAR10
from composer.datasets.coco import StreamingCOCO
from composer.datasets.imagenet import StreamingImageNet1k
from composer.datasets.streaming import StreamingDataset
from composer.datasets.streaming.download import download_or_wait
from composer.datasets.utils import pil_image_collate
from tests.datasets.test_streaming import get_fake_samples_decoders, write_synthetic_streaming_dataset


def get_dataset(name: str,
                local: str,
                split: str,
                shuffle: bool,
                batch_size: Optional[int],
                other_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[int, StreamingDataset]:
    other_kwargs = {} if other_kwargs is None else other_kwargs
    dataset_map = {
        'ade20k': {
            'remote': 's3://mosaicml-internal-dataset-ade20k/mds/1/',
            'num_samples': {
                'train': 20206,
                'val': 2000,
            },
            'class': StreamingADE20k,
            'kwargs': {},
        },
        'ade20k_sftp': {
            'remote':
                'sftp://mosaicml-test@s-d26bfe922c2141cca.server.transfer.us-west-2.amazonaws.com:22/mosaicml-internal-dataset-ade20k/mds/1',
            'num_samples': {
                'train': 20206,
                'val': 2000,
            },
            'class':
                StreamingADE20k,
            'kwargs': {},
        },
        'imagenet1k': {
            'remote': 's3://mosaicml-internal-dataset-imagenet1k/mds/1/',
            'num_samples': {
                'train': 1281167,
                'val': 50000,
            },
            'class': StreamingImageNet1k,
            'kwargs': {},
        },
        'coco': {
            'remote': 's3://mosaicml-internal-dataset-coco/mds/1/',
            'num_samples': {
                'train': 117266,
                'val': 4952,
            },
            'class': StreamingCOCO,
            'kwargs': {},
        },
        'c4': {
            'remote': 's3://mosaicml-internal-dataset-c4/mds/1/',
            'num_samples': {
                'train': 364868892,
                'val': 364608,
            },
            'class': StreamingC4,
            'kwargs': {
                'tokenizer_name': 'bert-base-uncased',
                'max_seq_len': 512
            },
        },
        'cifar10': {
            'remote': 's3://mosaicml-internal-dataset-cifar10/mds/1/',
            'num_samples': {
                'train': 50000,
                'val': 10000,
            },
            'class': StreamingCIFAR10,
            'kwargs': {},
        },
        'test_streaming_upload': {
            'remote': 's3://streaming-upload-test-bucket/',
            'num_samples': {
                'all': 0,
            },
            'class': StreamingDataset,
            'kwargs': {},
        }
    }
    if name not in dataset_map and split not in dataset_map[name]['num_samples'][split]:
        raise ValueError('Could not load dataset with name={name} and split={split}')

    d = dataset_map[name]
    expected_samples = d['num_samples'][split]
    remote = d['remote']
    kwargs = {**d['kwargs'], **other_kwargs}
    dataset = d['class'](remote=remote, local=local, split=split, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return (expected_samples, dataset)


@pytest.mark.remote
@pytest.mark.xfail(reason='Test is broken. See https://mosaicml.atlassian.net/browse/CO-762')
def test_upload_streaming_dataset(tmp_path: pathlib.Path, s3_bucket: str):
    num_samples = 1000
    original_path = str(tmp_path / 'original')
    download_path = str(tmp_path / 'downloaded')
    samples, decoders = get_fake_samples_decoders(num_samples)
    write_synthetic_streaming_dataset(original_path, samples, shard_size_limit=1 >> 16, upload=f's3://{s3_bucket}/')
    dataset = get_dataset('test_streaming_upload', download_path, 'all', False, 1, other_kwargs={'decoders': decoders})
    measured_samples = 0
    for _ in dataset:
        measured_samples += 1

    assert dataset[0] == measured_samples and measured_samples == num_samples


@pytest.mark.remote()
@pytest.mark.filterwarnings(r'ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('name', [
    'ade20k',
    'ade20k_sftp',
    'imagenet1k',
    'coco',
    'cifar10',
])
@pytest.mark.parametrize('split', ['val'])
@pytest.mark.xfail(reason='Test is broken. See https://mosaicml.atlassian.net/browse/CO-762')
def test_streaming_remote_dataset(tmp_path: pathlib.Path, name: str, split: str) -> None:

    # Build StreamingDataset
    build_start = time.time()
    expected_samples, dataset = get_dataset(name=name, local=str(tmp_path), split=split, shuffle=False, batch_size=None)
    build_end = time.time()
    build_dur = build_end - build_start
    print('Built dataset')

    # Test basic iteration
    rcvd_samples = 0
    iter_start = time.time()
    for _ in dataset:
        rcvd_samples += 1

        if (rcvd_samples % 1000 == 0):
            print(f'samples read: {rcvd_samples}')

    iter_end = time.time()
    iter_dur = iter_end - iter_start
    samples_per_sec = rcvd_samples / iter_dur

    # Print debug info
    print(f'build_dur={build_dur:.2f}s, iter_dur={iter_dur:.2f}, samples_per_sec={samples_per_sec:.2f}')

    # Test all samples arrived
    assert rcvd_samples == expected_samples


@pytest.mark.remote()
@pytest.mark.filterwarnings(r'ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('name', [
    'ade20k',
    'imagenet1k',
    'coco',
    'cifar10',
    'c4',
])
@pytest.mark.parametrize('split', ['val'])
@pytest.mark.xfail(reason='Test is broken. See https://mosaicml.atlassian.net/browse/CO-762')
def test_streaming_remote_dataloader(tmp_path: pathlib.Path, name: str, split: str) -> None:
    # Transformers imports required for batch collating
    pytest.importorskip('transformers')
    from transformers import DataCollatorForLanguageModeling
    from transformers.tokenization_utils_base import BatchEncoding

    # Data loading info
    shuffle = True
    batch_size = 8
    num_workers = 8
    drop_last = False
    persistent_workers = True

    # Build StreamingDataset
    ds_build_start = time.time()
    expected_samples, dataset = get_dataset(name=name,
                                            local=str(tmp_path),
                                            split=split,
                                            shuffle=shuffle,
                                            batch_size=batch_size)
    ds_build_end = time.time()
    ds_build_dur = ds_build_end - ds_build_start
    print('Built dataset')

    # Get collate_fn if needed
    collate_fn = None
    if name in ['ade20k', 'imagenet1k']:
        collate_fn = pil_image_collate
    elif name in ['c4']:
        if isinstance(dataset, StreamingC4):
            collate_fn = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=True, mlm_probability=0.15)
        else:
            raise ValueError('Expected dataset to be instance of StreamingC4')

    # Build DataLoader
    loader_build_start = time.time()
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        persistent_workers=persistent_workers,
                        collate_fn=collate_fn)
    loader_build_end = time.time()
    loader_build_dur = loader_build_end - loader_build_start

    # Print debug info
    print(f'ds_build_dur={ds_build_dur:.2f}s, loader_build_dur={loader_build_dur:.2f}s')

    for epoch in range(3):
        rcvd_samples = 0
        last_marker = 0
        marker_interval = len(dataset) // 20
        epoch_start = time.time()
        for _, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                n_samples = batch[0].shape[0]
            elif isinstance(batch, dict):
                first_key = list(batch.keys())[0]
                n_samples = batch[first_key].shape[0]
            elif isinstance(batch, BatchEncoding):
                first_key = list(batch.data.keys())[0]
                n_samples = batch.data[first_key].shape[0]
            else:
                raise ValueError(f'Unsure how to count n_samples for batch of type {type(batch)}')
            assert isinstance(n_samples, int)
            rcvd_samples += n_samples
            if rcvd_samples - last_marker > marker_interval:
                print(f'samples read: {rcvd_samples}')
                last_marker = rcvd_samples
        epoch_end = time.time()
        epoch_dur = epoch_end - epoch_start
        samples_per_sec = rcvd_samples / epoch_dur
        print(f'Epoch {epoch}: epoch_dur={epoch_dur:.2f}s, samples_per_sec={samples_per_sec:.2f}')

        # Test all samples arrived
        assert rcvd_samples == expected_samples


def test_download_from_http(httpserver: pytest_httpserver.HTTPServer, tmp_path: pathlib.Path):
    httpserver.expect_request('/data').respond_with_data('hi')
    local_path = str(tmp_path / 'data')
    download_or_wait(httpserver.url_for('/data'), local_path, wait=False)
    with open(local_path, 'r') as f:
        assert f.read() == 'hi'
