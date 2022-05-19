# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.models.efficientnetb0.efficientnets import EfficientNet


@pytest.mark.gpu
def test_efficientb0_activate_shape():
    # Running this test on cuda as convolutions are slow on CPU
    random_input = torch.rand(2, 3, 224, 224).cuda()

    model = EfficientNet.get_model_from_name(
        'efficientnet-b0',
        num_classes=1000,
        drop_connect_rate=0.2,
    ).cuda()
    # Test Stem
    out = model.conv_stem(random_input)
    out = model.bn1(out)
    out = model.act1(out)
    assert out.shape == (2, 32, 112, 112)

    # Test each block, shapes found at Table 1 of EfficientNet paper
    block_act_shape = [
        (2, 16, 112, 112),
        (2, 24, 56, 56),
        (2, 24, 56, 56),
        (2, 40, 28, 28),
        (2, 40, 28, 28),
        (2, 80, 14, 14),
        (2, 80, 14, 14),
        (2, 80, 14, 14),
        (2, 112, 14, 14),
        (2, 112, 14, 14),
        (2, 112, 14, 14),
        (2, 192, 7, 7),
        (2, 192, 7, 7),
        (2, 192, 7, 7),
        (2, 192, 7, 7),
        (2, 320, 7, 7),
    ]
    for i, block in enumerate(model.blocks):
        out = block(out)
        assert out.shape == block_act_shape[i]

    out = model.conv_head(out)
    assert out.shape == (2, 1280, 7, 7)
