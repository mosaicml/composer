# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp
from torchvision import datasets, transforms

from composer.datasets.hparams import DataloaderSpec, DatasetHparams


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

    def initialize_object(self) -> DataloaderSpec:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(
            self.datadir,
            train=self.is_train,
            download=self.download,
            transform=transform,
        )
        return DataloaderSpec(
            dataset=dataset,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
        )
