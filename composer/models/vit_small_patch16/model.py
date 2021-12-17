# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

from composer.models.base import MosaicClassifier
from composer.models.model_hparams import Initializer


class ViTSmallPatch16(MosaicClassifier):
    """Implements a ViT-S/16 wrapper around a MosaicClassifier.

    See this `paper <https://arxiv.org/pdf/2012.12877.pdf>` for details on ViT-S/16.

    Args:
        num_classes (int): The number of classes for the model.
    """
    def __init__(self, num_classes: int,
                 initializers: Optional[List[Initializer]] = None,
                 ) -> None:
        if initializers is None:
            initializers = [] # not using initializers right now
        from vit_pytorch import ViT
        model = ViT(image_size=224,
                    channels=3,
                    num_classes=num_classes,
                    dim=384,  # embed dim/width
                    patch_size=16,
                    depth=12,  # layers
                    heads=6,
                    mlp_dim=1536,
                    dropout=0.1
                    )
        super().__init__(module=model)
