# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper utilities to create FFCV datasets."""

import logging
import os
import sys
import textwrap
from argparse import ArgumentParser
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from tqdm import tqdm

from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.streaming import StreamingDataset

log = logging.getLogger(__name__)


def _get_parser():
    parser = ArgumentParser(description='Utility for converting datasets to ffcv format.')

    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'imagenet1k'],
                        help=textwrap.dedent("""\
                                Dataset to use. Default: cifar10"""))
    parser.add_argument('--remote',
                        type=str,
                        help=textwrap.dedent("""\
                                Remote directory (S3 or local filesystem) where dataset is stored., Example: s3://my-s3-bucket-name"""
                                            ))
    parser.add_argument('--local',
                        type=str,
                        default=None,
                        help=textwrap.dedent("""\
                                Local filesystem directory where dataset is cached during operation. Default: None"""))
    parser.add_argument('--split',
                        type=str,
                        default='train',
                        choices=['train', 'val'],
                        help=textwrap.dedent("""\
                                Split to use. Default: train"""))

    parser.add_argument('--datadir',
                        type=str,
                        default=None,
                        help=textwrap.dedent("""\
                                Location of the dataset. Default: None"""))

    parser.add_argument('--download',
                        type=bool,
                        default=False,
                        help=textwrap.dedent("""\
                                Download the dataset if possible. Default: False"""))

    parser.add_argument('--write_path',
                        type=str,
                        default=None,
                        help=textwrap.dedent("""\
                                File path to use for writing the dataset. Default: /tmp/<dataset>_<split>.ffcv"""))

    parser.add_argument('--write_mode',
                        type=str,
                        default='proportion',
                        choices=['raw', 'jpg', 'smart', 'proportion'],
                        help=textwrap.dedent("""\
                                Write mode to use. raw is uint8 values, jpg is jpeg compressed images, smart is
                                compressing based on image size and proportion is according to the given
                                compress_probability. Default: proportion"""))

    parser.add_argument('--max_resolution', type=int, default=500, help='Max resoultion for images.')

    parser.add_argument('--num_workers', type=int, default=64, help='Number of workers to use.')

    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size to use.')

    parser.add_argument('--jpeg_quality', type=int, default=90, help='Quality of jpeg.')

    parser.add_argument('--subset', type=int, default=-1, help='Only use a subset of dataset.')

    parser.add_argument('--compress_probability',
                        type=float,
                        required=False,
                        default=0.50,
                        help='Compress the given fraction of images to jpeg while writing the ffcv dataset.')
    return parser


def _parse_args():
    parser = _get_parser()

    args = parser.parse_args()

    if args.datadir is not None:
        log.info(f'Will read from local directory: {args.datadir}.')
    else:
        if args.local is None:
            args.local = f'/tmp/mds-cache/mds-{args.dataset}/'

        if args.remote.startswith('s3://'):
            log.info(f'Will read from remote: {args.remote}.')
        else:
            log.info(f'Will read from local: {args.remote}.')

    if args.write_path is None:
        args.write_path = f'/tmp/{args.dataset}_{args.split}.ffcv'

    if os.path.exists(args.write_path):
        log.error(f'Destination already exists: {args.write_path}')
        sys.exit(-1)

    return args


def cache_streaming(dataset, num_workers=64):
    """A function to iterate over all the samples in the dataset to cache it locally."""
    if 'LOCAL_WORLD_SIZE' not in os.environ:
        os.environ['LOCAL_WORLD_SIZE'] = '1'

    def _null_collate_fn(batch):
        return None

    dl = torch.utils.data.DataLoader(dataset,
                                     batch_size=512,
                                     num_workers=num_workers,
                                     pin_memory=False,
                                     drop_last=False,
                                     collate_fn=_null_collate_fn)

    # empty loop to download and cache the dataset
    print('download and cache dataset')
    for _ in tqdm(dl):
        pass


class PILImageClassStreamingDataset(StreamingDataset):
    """A subclass of StreamingDataset that asserts and returns tuples (PIL.Image, np.int64).

    Intended to enforce sample types for pipelines that expect the tuple format (image, class).
    """

    def __getitem__(self, idx: int) -> Tuple[Image.Image, np.int64]:
        obj = super().__getitem__(idx)
        assert 'x', 'y' in obj.keys()
        assert isinstance(obj['x'], Image.Image)
        assert isinstance(obj['y'], np.int64)
        return (obj['x'], obj['y'])


def _main():

    args = _parse_args()

    ds = None
    if args.datadir is not None:
        if args.dataset == 'cifar10':
            ds = CIFAR10(root=args.datadir, train=(args.split == 'train'), download=args.download)
        elif args.dataset == 'imagenet1k':
            ds = ImageFolder(os.path.join(args.datadir, args.split))
        else:
            raise ValueError(f'Unsupported dataset: {args.dataset}. Checkout the list of supported datasets with -h')

        if args.subset > 0:
            ds = Subset(ds, range(args.subset))
    else:
        remote = os.path.join(args.remote, args.split)
        local = os.path.join(args.local, args.split)

        # Build a simple StreamingDataset that reads in the raw images, no processing
        # Differet datasets are created + encoded differently, so that's why we need different decoders
        if args.dataset == 'cifar10':

            def _decode_image(data):
                arr = np.frombuffer(data, np.uint8)
                arr = arr.reshape(32, 32, 3)
                return Image.fromarray(arr)

            def _decode_class(data):
                return np.frombuffer(data, dtype=np.int64)[0]
        elif args.dataset == 'imagenet1k':

            def _decode_image(data):
                return Image.open(BytesIO(data)).convert('RGB')

            def _decode_class(data):
                return np.frombuffer(data, dtype=np.int64)[0]

        decoders = {'x': _decode_image, 'y': _decode_class}
        dataset = PILImageClassStreamingDataset(remote=remote, local=local, decoders=decoders, shuffle=False)

        # Iterate over the dataset and cache it, so it can be used for random access
        cache_streaming(dataset=dataset, num_workers=args.num_workers)

        write_ffcv_dataset(dataset=dataset,
                           write_path=args.write_path,
                           max_resolution=args.max_resolution,
                           num_workers=args.num_workers,
                           write_mode=args.write_mode,
                           compress_probability=args.compress_probability,
                           jpeg_quality=args.jpeg_quality,
                           chunk_size=args.chunk_size)


if __name__ == '__main__':
    sys.exit(_main())
