# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
import torch

from composer.algorithms.factorize import FactorizedConv2d


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('h', [5])
@pytest.mark.parametrize('w', [6])
@pytest.mark.parametrize('in_channels', [4, 8])
@pytest.mark.parametrize('out_channels', [4, 8])
@pytest.mark.parametrize('kernel_size', [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1), (5, 5)])
def test_shapes(batch_size, h, w, in_channels, out_channels, kernel_size, **kwargs):
    X = torch.randn(batch_size, in_channels, h, w)
    conv = FactorizedConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs)
    Y = conv(X)

    assert Y.ndim == 4
    assert Y.shape[:2] == (batch_size, out_channels)


def test_update_layer_twice():
    batch_size = 2
    h = 5
    w = 6
    C_in = 32
    C_out = 32
    C_latent = 32
    X = torch.randn(batch_size, C_in, h, w)
    kernel_size = (3, 3)
    module = FactorizedConv2d(in_channels=C_in,
                              out_channels=C_out,
                              latent_channels=C_latent,
                              kernel_size=kernel_size,
                              padding=0)
    assert module.conv1 is None  # initially not factorized

    def _check_conv_shapes(module: FactorizedConv2d, C_in, C_out, C_latent):
        assert module.latent_channels == C_latent
        assert module.conv0.in_channels == C_in
        assert module.conv0.out_channels == C_latent
        assert module.conv1 is not None
        assert module.conv1.in_channels == C_latent
        assert module.conv1.out_channels == C_out

    module.set_rank(X, 24)
    _check_conv_shapes(module, C_in=C_in, C_out=C_out, C_latent=24)
    module.set_rank(X, 16)
    _check_conv_shapes(module, C_in=C_in, C_out=C_out, C_latent=16)
