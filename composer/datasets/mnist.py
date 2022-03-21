# Copyright 2021 MosaicML. All Rights Reserved.

"""MNIST image classification dataset.

The MNIST dataset is a collection of labeled 28x28 black and white images of handwritten examples of the numbers 0-9.
See the `wikipedia entry <https://en.wikipedia.org/wiki/MNIST_database>`_ for more details.
"""

from dataclasses import dataclass

import yahp as hp
from torchvision import datasets, transforms

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist

__all__ = ["MNISTDatasetHparams", "MNISTWebDatasetHparams"]


@dataclass
class MNISTDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the MNIST dataset for image classification.

    Args:
        download (bool, optional): Whether to download the dataset, if needed. Default:
            ``True``.
    """
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader:
        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=60_000 if self.is_train else 10_000,
                data_shape=[1, 28, 28],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        else:
            if self.datadir is None:
                raise ValueError("datadir is required if synthetic is False")

            transform = transforms.Compose([transforms.ToTensor()])
            dataset = datasets.MNIST(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transform,
            )
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        return dataloader_hparams.initialize_object(dataset=dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)


@dataclass
class MNISTWebDatasetHparams(WebDatasetHparams):
    """Defines an instance of the MNIST WebDataset for image classification.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
        name (str): Key used to determine where dataset is cached on local filesystem.
    """

    remote: str = hp.optional('WebDataset S3 bucket name', default='s3://mosaicml-internal-dataset-mnist')
    name: str = hp.optional('WebDataset local cache name', default='mnist')

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader:
        from composer.datasets.webdataset import load_webdataset

        split = 'train' if self.is_train else 'val'
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        preprocess = lambda dataset: dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
        dataset = load_webdataset(self.remote, self.name, split, self.webdataset_cache_dir,
                                  self.webdataset_cache_verbose, self.shuffle, self.shuffle_buffer, preprocess,
                                  dist.get_world_size(), dataloader_hparams.num_workers, batch_size, self.drop_last)
        return dataloader_hparams.initialize_object(dataset=dataset,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    drop_last=self.drop_last)
