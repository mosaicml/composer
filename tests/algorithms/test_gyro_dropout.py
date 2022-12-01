# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Written by Gihyun Park, Junyeol Lee, and Jiwon Seo

import torch

from composer.algorithms.gyro_dropout import GyroDropoutLayer


def test_gyro_dropout_masking():
    batch_size = 256
    output_feature = 512
    x = torch.randn(batch_size, output_feature)

    dropout_layer = GyroDropoutLayer(
        iters_per_epoch=196,
        max_epoch=100,
        p=0.5,
        sigma=256,
        tau=16,
    )
    y = dropout_layer(x)

    mask = dropout_layer.dropout_mask
    p = dropout_layer.p
    for i in range(batch_size):
        for j in range(output_feature):
            assert x[i][j] * mask[i][j] * (1 / (1 - p)) == y[i][j]


def test_gyro_dropout_mask_pattern():
    batch_size = 256
    output_feature = 512
    x = torch.randn(batch_size, output_feature)

    dropout_layer = GyroDropoutLayer(
        iters_per_epoch=196,
        max_epoch=100,
        p=0.5,
        sigma=256,
        tau=16,
    )
    _ = dropout_layer(x)

    mask = dropout_layer.dropout_mask
    tau = dropout_layer.tau
    pivot = 0
    for i in range(output_feature):
        for j in range(batch_size):
            if j % tau == 0:
                pivot = mask[j][i]
            else:
                assert pivot == mask[j][i]
