# Copyright 2021 MosaicML. All Rights Reserved.
from composer.models.base import ComposerClassifier

__all__ = ["ViTSmallPatch16"]


class ViTSmallPatch16(ComposerClassifier):
    """Implements a ViT-S/16 wrapper around a MosaicClassifier.

    See this `paper <https://arxiv.org/pdf/2012.12877.pdf>` for details on ViT-S/16.

    Args:
        num_classes (int, optional): number of classes for the model. Default: 1000.
        image_size (int, optional): input image size. If you have rectangular images, make sure your image size is the maximum of the width and height. Default: 224.
        channels (int, optional): number of  image channels. Default: 3.
        dropout (float, optional): 0.0 - 1.0 dropout rate. Default: 0.
        embedding_dropout (float, optional): 0.0 - 1.0 embedding dropout rate. Default: 0.
    """

    def __init__(self,
                 num_classes: int = 1000,
                 image_size: int = 224,
                 channels: int = 3,
                 dropout: float = 0.0,
                 embedding_dropout: float = 0.0) -> None:
        from vit_pytorch import ViT
        model = ViT(
            image_size=image_size,
            channels=channels,
            num_classes=num_classes,
            dim=384,  # embed dim/width
            patch_size=16,
            depth=12,  # layers
            heads=6,
            mlp_dim=1536,
            dropout=dropout,
            emb_dropout=embedding_dropout)
        super().__init__(module=model)
