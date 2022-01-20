# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp
from torchvision import datasets, transforms

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist


@dataclass
class MNISTDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the MNIST dataset for image classification.

    Parameters:
        download (bool): Whether to download the dataset, if needed.
    """
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
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
