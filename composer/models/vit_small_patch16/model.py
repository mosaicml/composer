# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implements ViT-S/16 as a :class:`.ComposerClassifier`."""

from composer.models.tasks import ComposerClassifier

__all__ = ['vit_small_patch16']


def vit_small_patch16(num_classes: int = 1000,
                      image_size: int = 224,
                      channels: int = 3,
                      dropout: float = 0.0,
                      embedding_dropout: float = 0.0):
    """Helper function to create a :class:`.ComposerClassifier` using a ViT-S/16 model.

    See `Training data-efficient image transformers & distillation through attention <https://arxiv.org/pdf/2012.12877.pdf>`_
        (Touvron et al, 2021) for details on ViT-S/16.

    Args:
        num_classes (int, optional): number of classes for the model. Default: ``1000``.
        image_size (int, optional): input image size. If you have rectangular images, make sure your image
         size is the maximum of the width and height. Default: ``224``.
        channels (int, optional): number of  image channels. Default: ``3``.
        dropout (float, optional): 0.0 - 1.0 dropout rate. Default: ``0``.
        embedding_dropout (float, optional): 0.0 - 1.0 embedding dropout rate. Default: ``0``.

    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with a ViT-S/16 model.
    """

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

    composer_model = ComposerClassifier(module=model)
    return composer_model
