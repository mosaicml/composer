# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.base import MosaicClassifier


class ViTSmallPatch16(MosaicClassifier):
    """Implements a ViT-S/16 wrapper around a MosaicClassifier.

    See this `paper <https://arxiv.org/pdf/2012.12877.pdf>` for details on ViT-S/16.

    Args:
        image_size (int): input image size, assumed to be square. example: 224 for (224, 224) sized inputs.
        num_channels (int): The number of input channels.
        num_classes (int): The number of classes for the model.
    """
    def __init__(self, num_classes: int):
        from vit_pytorch import ViT
        model = ViT(image_size=224,
                    channels=3,
                    num_classes=num_classes,
                    dim=384,  # embed dim/width
                    patch_size=16,
                    depth=12,  # layers
                    heads=6,
                    mlp_dim=1536,
                    )
        super().__init__(module=model)
