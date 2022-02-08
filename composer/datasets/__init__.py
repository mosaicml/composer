# Copyright 2021 MosaicML. All Rights Reserved.

"""Datasets  TODO -- more description.

:class:`DataloaderHparams` contains the :class:`torch.utils.data.dataloader` settings that are common across both training and eval datasets:

* ``num_workers``
* ``prefetch_factor``
* ``persistent_workers``
* ``pin_memory``
* ``timeout``

Each :class:`DatasetHparams` is then responsible for settings such as:

* ``dataset``
* ``drop_last``
* ``shuffle``
* ``collate_fn``

A :class:`DatasetHparams` is responsible for returning a :class:`torch.utils.data.dataloader` or a :class:`DataloaderSpec`.
"""
from composer.datasets.ade20k import ADE20kDatasetHparams as ADE20kDatasetHparams
from composer.datasets.brats import BratsDatasetHparams as BratsDatasetHparams
from composer.datasets.cifar10 import CIFAR10DatasetHparams as CIFAR10DatasetHparams
from composer.datasets.dataloader import DataloaderHparams as DataloaderHparams
from composer.datasets.dataloader import WrappedDataLoader as WrappedDataLoader
from composer.datasets.dataset_registry import get_dataset_registry as get_dataset_registry
from composer.datasets.evaluator import EvaluatorHparams as EvaluatorHparams
from composer.datasets.glue import GLUEHparams as GLUEHparams
from composer.datasets.hparams import DatasetHparams as DatasetHparams
from composer.datasets.hparams import SyntheticHparamsMixin as SyntheticHparamsMixin
from composer.datasets.imagenet import ImagenetDatasetHparams as ImagenetDatasetHparams
from composer.datasets.lm_datasets import LMDatasetHparams as LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams as MNISTDatasetHparams
from composer.datasets.synthetic import MemoryFormat as MemoryFormat
from composer.datasets.synthetic import SyntheticBatchPairDataset as SyntheticBatchPairDataset
from composer.datasets.synthetic import SyntheticDataLabelType as SyntheticDataLabelType
from composer.datasets.synthetic import SyntheticDataType as SyntheticDataType
from composer.datasets.synthetic import SyntheticPILDataset as SyntheticPILDataset
