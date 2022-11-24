# Copyright 2022  Gihyun Park, Junyeol Lee, and Jiwon Seo
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.algorithms.gyro_dropout import GyroDropout, apply_gyro_dropout
from composer.models import ComposerClassifier


class DropoutLayer(ComposerClassifier):
    def __init__(self) -> None:
        dropout = torch.nn.Dropout(0.5)
        net = torch.nn.Sequential(dropout)
        super().__init__(module=net)

        self.dropout = dropout


@pytest.fixture
def gyro_dropout_layer() -> ComposerClassifier:
    model = DropoutLayer()
    apply_gyro_dropout(model=model, iters_per_epoch=196, max_epoch=100, p=0.5, sigma=256, tau=16)
    return model


def test_gyro_dropout_masking(gyro_dropout_layer: torch.nn.Module):
    batch_size = 256
    output_feature = 512
    x = torch.randn(batch_size, output_feature)

    model = gyro_dropout_layer
    y = model((x, None))
    
    mask = model.dropout.dropout_mask
    p = model.dropout.p
    for i in range(batch_size):
        for j in range(output_feature):
            assert x[i][j]*mask[i][j]*(1/(1-p)) == y[i][j]


def test_gyro_dropout_mask_pattern(gyro_dropout_layer: torch.nn.Module):
    batch_size = 256
    output_feature = 512
    x = torch.randn(batch_size, output_feature)

    model = gyro_dropout_layer
    y = model((x, None))
    
    mask = model.dropout.dropout_mask
    tau = model.dropout.tau
    pivot = 0
    for i in range(output_feature):
        for j in range(batch_size):
            if j % tau == 0:
                pivot = mask[j][i]
            else:
                assert pivot == mask[j][i]
