# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

from composer.datasets.photo_classification import PhotoClassificationDatasetHparams


@dataclass
class PlacesDatasetHparams(PhotoClassificationDatasetHparams):
    """Defines an instance of the Places365 small dataset for scene classification.

    Parameters:
        resize_size (int, optional): The resize size to use. Defaults to -1 to not resize.
        crop size (int): The crop size to use.
    """
    pass
