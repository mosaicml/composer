# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
MosaicML Streaming Datasets for cloud-native model training.

This is a new dataset class :class:`StreamingDataset(torch.utils.data.IterableDatset)`
and associated dataset format: ``shard-[00x].mds`` that has much better performance, shuffling, and usability than existing solutions.

A brief list of improvements:

* No requirement of ``n_samples % n_shards == 0``: Sharded datasets are complete with no dropped samples.
* No requirement of ``n_shards % n_cpu_workers == 0``: Supports reading from any # of devices, with any # of CPU workers.
* Dataset is downloaded only ~once, regardless of # nodes and # devices and # CPU workers, no duplicate downloads and egress fees.
* Dataset is cached on local storage after epoch 1.
* When used with a :class:`torch.utils.data.DataLoader`, the epoch boundaries are consistent (# samples, # batches) regardless of ``num_workers``, producing (nearly) the same behavior as a map-style :class:`torch.utils.data.Dataset`.
* When data is read from a single device with ``num_workers <= 1``, samples are read in-order (useful for local dataset inspection).
* Dataset order is deterministic for a given value of ``world_size`` and ``num_workers``.
* Dataset can resume mid epoch without needing to "spin" through previous values.
* (TODO) Supports lazy random-access retrieval of samples (useful for local dataset inspection).
"""

from composer.datasets.streaming.dataset import StreamingDataset
from composer.datasets.streaming.writer import StreamingDatasetWriter

__all__ = ['StreamingDataset', 'StreamingDatasetWriter']
