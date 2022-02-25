# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
import textwrap
from dataclasses import dataclass
from typing import Optional, Union

from torchvision import transforms

try:
    import custom_inherit
except ImportError:
    # if custom_inherit is not installed, then the docstrings will be incomplete. That's fine.
    metaclass = abc.ABCMeta
else:
    metaclass = custom_inherit.DocInheritMeta(style="google_with_merge", abstract_base_class=True)

import yahp as hp

from composer.core.types import DataLoader, DataSpec, MemoryFormat
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.webdataset import load_webdataset, size_webdataset
from composer.utils import dist


@dataclass
class SyntheticHparamsMixin(hp.Hparams, abc.ABC):
    """Synthetic dataset parameter mixin for :class:`DatasetHparams`.

    Parameters:
        use_synthetic (bool, optional): Whether to use synthetic data. (Default: ``False``)
        synthetic_num_unique_samples (int, optional): The number of unique samples to allocate memory for.
            Ignored if :attr:`use_synthetic` is False. (Default: ``100``)
        synthetic_device (str, optonal): The device to store the sample pool.
            Set to ``cuda`` to store samples on the GPU and eliminate PCI-e bandwidth with the dataloader.
            Set to ``cpu`` to move data between host memory and the device on every batch.
            Ignored if :attr:`use_synthetic` is False. (Default: ``cpu``)
        synthetic_memory_format: The :class:`MemoryFormat` to use.
            Ignored if :attr:`use_synthetic` is False. (Default: ``CONTIGUOUS_FORMAT``)
    """

    use_synthetic: bool = hp.optional("Whether to use synthetic data. Defaults to False.", default=False)
    synthetic_num_unique_samples: int = hp.optional("The number of unique samples to allocate memory for.", default=100)
    synthetic_device: str = hp.optional("Device to store the sample pool. Should be `cuda` or `cpu`. Defauls to `cpu`.",
                                        default="cpu")
    synthetic_memory_format: MemoryFormat = hp.optional("Memory format. Defaults to contiguous format.",
                                                        default=MemoryFormat.CONTIGUOUS_FORMAT)


@dataclass
class DatasetHparams(hp.Hparams, abc.ABC, metaclass=metaclass):
    """Abstract base class for hyperparameters to initialize a dataset.

    Parameters:
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data (the default) or validation data.
        drop_last (bool):
            If the number of samples is not divisible by the batch size, whether
            to drop the last batch (the default) or pad the last batch with zeros.
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.
    """

    is_train: bool = hp.optional("Whether to load the training data (the default) or validation data.", default=True)
    drop_last: bool = hp.optional(textwrap.dedent("""If the number of samples is not divisible by the batch size,
        whether to drop the last batch (the default) or pad the last batch with zeros."""),
                                  default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch. Defaults to True.", default=True)

    datadir: Optional[str] = hp.optional("The path to the data directory", default=None)

    @abc.abstractmethod
    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataSpec]:
        """Creates a :class:`DataLoader` or :class:`DataloaderSpec` for this dataset.

        Parameters:
            batch_size (int): The size of the batch the dataloader should yield. This batch size is
                device-specific and already incorporates the world size.
            dataloader_hparams (DataloaderHparams): The dataset-independent hparams for the dataloader

        Returns:
            Dataloader or DataSpec: The dataloader, or if the dataloader yields batches of custom types,
            a :class:`DataSpec`.
        """
        pass


@dataclass
class WebDatasetHparams(DatasetHparams, abc.ABC, metaclass=metaclass):
    """Abstract base class for hyperparameters to initialize a dataset.

    Parameters:
        webdataset_cache_dir (str): WebDataset cache directory.
        webdataset_cache_verbose (str): WebDataset cache verbosity.
    """

    webdataset_cache_dir: str = hp.optional('WebDataset cache directory', default='/tmp/webdataset_cache/')
    webdataset_cache_verbose: bool = hp.optional('WebDataset cache verbosity', default=False)

    @abc.abstractmethod
    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> Union[DataLoader, DataSpec]:
        """Creates a :class:`DataLoader` or :class:`DataloaderSpec` for this dataset.

        Parameters:
            batch_size (int): The size of the batch the dataloader should yield. This batch size is
                device-specific and already incorporates the world size.
            dataloader_hparams (DataloaderHparams): The dataset-independent hparams for the dataloader

        Returns:
            Dataloader or DataSpec: The dataloader, or if the dataloader yields batches of custom types,
            a :class:`DataSpec`.
        """
        pass


@dataclass
class JpgClsWebDatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Common functionality for (jpg, cls) WebDatasets.

    Parameters:
        dataset_s3_bucket (str): S3 bucket or root directory where dataset is stored.
        dataset_name (str): Key used to determine where dataset is cached on local filesystem.
        n_train_samples (int): Number of training samples.
        n_val_samples (int): Number of validation samples.
        height (int): Sample image height in pixels.
        width (int): Sample image width in pixels.
        n_classes (int): Number of output classes.
        channel_means (list of float): Channel means for normalization.
        channel_stds (list of float): Channel stds for normalization.
    """

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=self.n_train_samples if self.is_train else self.n_val_samples,
                data_shape=[3, self.height, self.width],
                num_classes=self.n_classes,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
        else:
            if self.is_train:
                split = 'train'
                transform = transforms.Compose([
                    transforms.RandomCrop((self.height, self.width), (self.height // 8, self.width // 8)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.channel_means, self.channel_stds),
                ])
            else:
                split = 'val'
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.channel_means, self.channel_stds),
                ])
            dataset, meta = load_webdataset(self.dataset_s3_bucket, self.dataset_name, split, self.webdataset_cache_dir,
                                            self.webdataset_cache_verbose)
            if self.shuffle:
                dataset = dataset.shuffle(512)
            dataset = dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
            dataset = size_webdataset(dataset, meta['n_shards'], meta['samples_per_shard'], dist.get_world_size(),
                                      dataloader_hparams.num_workers, batch_size, self.drop_last)
        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    drop_last=self.drop_last)
