# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import numpy as np

from composer.core import Dataset
from composer.utils import MissingConditionalImportError

try:
    import ffcv
    ffcv_installed = True
except ImportError:
    ffcv_installed = False

log = logging.getLogger(__name__)

__all__ = ['write_ffcv_dataset', 'ffcv_monkey_patches']


def _require_ffcv():
    if not ffcv_installed:
        raise MissingConditionalImportError(extra_deps_group='ffcv', conda_package='ffcv')


def ffcv_monkey_patches():
    _require_ffcv()

    # ffcv's __len__ function is expensive as it always calls self.next_traversal_order which does shuffling.
    # Composer calls len(dataloader) function in training loop for every batch and thus len function causes 2x slowdown.
    # ffcv's __len__ is fixed in 1.0.0 branch but for another reason (https://github.com/libffcv/ffcv/issues/163).
    def new_len(self):
        if not hasattr(self, 'init_traversal_order'):
            self.init_traversal_order = self.next_traversal_order()
        if self.drop_last:
            return len(self.init_traversal_order) // self.batch_size
        else:
            return int(np.ceil(len(self.init_traversal_order) / self.batch_size))

    ffcv.loader.loader.Loader.__len__ = new_len


def write_ffcv_dataset(dataset: Optional[Dataset] = None,
                       write_path: str = '/tmp/dataset.ffcv',
                       max_resolution: Optional[int] = None,
                       num_workers: int = 16,
                       write_mode: str = 'raw',
                       compress_probability: float = 0.50,
                       jpeg_quality: float = 90,
                       chunk_size: int = 100):
    """Converts PyTorch compatible ``dataset`` into FFCV format at filepath ``write_path``.

    Args:
        dataset (Iterable[Sample]): A PyTorch dataset. Default: ``None``.
        write_path (str): Write results to this file. Default: ``"/tmp/dataset.ffcv"``.
        max_resolution (int): Limit resolution if provided. Default: ``None``.
        num_workers (int): Numbers of workers to use. Default: ``16``.
        write_mode (str): Write mode for the dataset. Default: ``'raw'``.
        compress_probability (float): Probability with which image is JPEG-compressed. Default: ``0.5``.
        jpeg_quality (float): Quality to use for jpeg compression. Default: ``90``.
        chunk_size (int): Size of chunks processed by each worker during conversion. Default: ``100``.
    """

    _require_ffcv()
    if dataset is None:
        raise ValueError('dataset should not be None.')

    log.info(f'Writing dataset in FFCV <file>.ffcv format to {write_path}.')
    writer = ffcv.writer.DatasetWriter(write_path, {
        'image':
            ffcv.fields.RGBImageField(write_mode=write_mode,
                                      max_resolution=max_resolution,
                                      compress_probability=compress_probability,
                                      jpeg_quality=jpeg_quality),
        'label':
            ffcv.fields.IntField()
    },
                                       num_workers=num_workers)
    writer.from_indexed_dataset(dataset, chunksize=chunk_size)
