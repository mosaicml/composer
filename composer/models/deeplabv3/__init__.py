# Copyright 2022 MosaicML. All Rights Reserved.

"""DeepLabV3 for image segmentation."""
from composer.models.deeplabv3.deeplabv3 import ComposerDeepLabV3 as ComposerDeepLabV3
from composer.models.deeplabv3.deeplabv3_hparams import DeepLabV3Hparams as DeepLabV3Hparams

__all__ = ["ComposerDeepLabV3", "DeepLabV3Hparams"]
