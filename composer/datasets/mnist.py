# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp
from torchvision import datasets, transforms

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.utils import ddp


@dataclass
class MNISTDatasetHparams(DatasetHparams):
    """Defines an instance of the MNIST dataset for image classification.
    
    Parameters:
        is_train (bool): Whether to load the training or validation dataset.
        datadir (str): Data directory to use.
        download (bool): Whether to download the dataset, if needed.
        drop_last (bool): Whether to drop the last samples for the last batch.
        shuffle (bool): Whether to shuffle the dataset for each epoch.
    """

    is_train: bool = hp.required("whether to load the training or validation dataset")
    datadir: str = hp.required("data directory")
    download: bool = hp.required("whether to download the dataset, if needed")
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(
            self.datadir,
            train=self.is_train,
            download=self.download,
            transform=transform,
        )
        sampler = ddp.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)
