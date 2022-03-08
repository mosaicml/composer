# Copyright 2021 MosaicML. All Rights Reserved.

"""Natively supported datasets."""

from composer.datasets.ade20k import ADE20kDatasetHparams as ADE20kDatasetHparams
from composer.datasets.ade20k import ADE20kWebDatasetHparams as ADE20kWebDatasetHparams
from composer.datasets.brats import BratsDatasetHparams as BratsDatasetHparams
from composer.datasets.c4 import C4DatasetHparams as C4DatasetHparams
from composer.datasets.cifar import CIFAR10DatasetHparams as CIFAR10DatasetHparams
from composer.datasets.cifar import CIFAR10WebDatasetHparams as CIFAR10WebDatasetHparams
from composer.datasets.cifar import CIFAR20WebDatasetHparams as CIFAR20WebDatasetHparams
from composer.datasets.cifar import CIFAR100WebDatasetHparams as CIFAR100WebDatasetHparams
from composer.datasets.coco import COCODatasetHparams as COCODatasetHparams
from composer.datasets.dataloader import DataloaderHparams as DataloaderHparams
from composer.datasets.dataloader import WrappedDataLoader as WrappedDataLoader
from composer.datasets.dataset_registry import get_dataset_registry as get_dataset_registry
from composer.datasets.evaluator import EvaluatorHparams as EvaluatorHparams
from composer.datasets.glue import GLUEHparams as GLUEHparams
from composer.datasets.hparams import DatasetHparams as DatasetHparams
from composer.datasets.hparams import SyntheticHparamsMixin as SyntheticHparamsMixin
from composer.datasets.hparams import WebDatasetHparams as WebDatasetHparams
from composer.datasets.imagenet import Imagenet1kWebDatasetHparams as Imagenet1kWebDatasetHparams
from composer.datasets.imagenet import ImagenetDatasetHparams as ImagenetDatasetHparams
from composer.datasets.imagenet import TinyImagenet200WebDatasetHparams as TinyImagenet200WebDatasetHparams
from composer.datasets.lm_datasets import LMDatasetHparams as LMDatasetHparams
from composer.datasets.mnist import MNISTDatasetHparams as MNISTDatasetHparams
from composer.datasets.mnist import MNISTWebDatasetHparams as MNISTWebDatasetHparams
from composer.datasets.synthetic import MemoryFormat as MemoryFormat
from composer.datasets.synthetic import SyntheticBatchPairDataset as SyntheticBatchPairDataset
from composer.datasets.synthetic import SyntheticDataLabelType as SyntheticDataLabelType
from composer.datasets.synthetic import SyntheticDataType as SyntheticDataType
from composer.datasets.synthetic import SyntheticPILDataset as SyntheticPILDataset
