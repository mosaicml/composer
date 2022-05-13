# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.algorithms.factorize import FactorizedConv2d, FactorizedLinear


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('h', [5])
@pytest.mark.parametrize('w', [6])
@pytest.mark.parametrize('in_channels', [4, 8])
@pytest.mark.parametrize('out_channels', [4, 8])
@pytest.mark.parametrize('kernel_size', [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1), (5, 5)])
def test_factorized_conv2d_shapes(batch_size, h, w, in_channels, out_channels, kernel_size):
    X = torch.randn(batch_size, in_channels, h, w)
    conv = FactorizedConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    Y = conv(X)

    assert Y.ndim == 4
    assert Y.shape[:2] == (batch_size, out_channels)


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('in_features', [7, 8, 11])
@pytest.mark.parametrize('out_features', [4, 6, 8])
def test_factorized_linear_shapes(batch_size, in_features, out_features):
    X = torch.randn(batch_size, in_features)
    module = FactorizedLinear(in_features=in_features, out_features=out_features)
    Y = module(X)

    assert Y.ndim == 2
    assert Y.shape == (batch_size, out_features)


def test_update_factorized_conv2d_twice():
    batch_size = 2
    h = 5
    w = 6
    C_in = 36
    C_out = 40
    C_latent = 16
    X = torch.randn(batch_size, C_in, h, w)
    kernel_size = (3, 3)
    module = FactorizedConv2d(in_channels=C_in,
                              out_channels=C_out,
                              latent_channels=C_latent,
                              kernel_size=kernel_size,
                              padding=0)

    def _check_conv_shapes(module: FactorizedConv2d, C_in, C_out, C_latent):
        assert module.latent_channels == C_latent
        assert module.module0 is not None
        assert module.module0.in_channels == C_in
        assert module.module0.out_channels == C_latent
        assert isinstance(module.module0.weight, torch.Tensor)
        assert module.module0.weight.shape[:2] == (C_latent, C_in)
        assert module.module1 is not None
        assert module.module1.in_channels == C_latent
        assert module.module1.out_channels == C_out
        assert isinstance(module.module1.weight, torch.Tensor)
        assert module.module1.weight.shape[:2] == (C_out, C_latent)

    for new_C_latent in [12, 8]:
        module.set_rank(X, new_C_latent)
        _check_conv_shapes(module, C_in=C_in, C_out=C_out, C_latent=new_C_latent)


def test_update_factorized_linear_twice():
    batch_size = 2
    d_in = 36
    d_out = 40
    d_latent = 16
    X = torch.randn(batch_size, d_in)
    module = FactorizedLinear(in_features=d_in, out_features=d_out, latent_features=d_latent)

    def _check_shapes(module: FactorizedLinear, d_in, d_out, d_latent):
        assert module.latent_features == d_latent
        assert module.module0.in_features == d_in
        assert module.module0.out_features == d_latent
        # linear layer weights have shape (out_features, in_features)
        assert module.module0.weight.shape == (d_latent, d_in)
        assert module.module1 is not None
        assert module.module1.in_features == d_latent
        assert module.module1.out_features == d_out
        assert module.module1.weight.shape == (d_out, d_latent)

    _check_shapes(module, d_in=d_in, d_out=d_out, d_latent=d_latent)
    for new_d_latent in [12, 8]:
        module.set_rank(X, new_d_latent)
        _check_shapes(module, d_in=d_in, d_out=d_out, d_latent=new_d_latent)


@pytest.mark.parametrize('sizes', [(8, 8, 4), (7, 5, 3), (10, 10, 0.5)])
def test_factorized_conv_init_throws_if_latent_too_big(sizes):
    with pytest.raises(ValueError):
        FactorizedConv2d(in_channels=sizes[0], out_channels=sizes[1], latent_channels=sizes[2], kernel_size=(1, 1))


@pytest.mark.parametrize('sizes', [(8, 8, 4), (7, 5, 3), (10, 10, 0.5)])
def test_factorized_linear_init_throws_if_latent_too_big(sizes):
    with pytest.raises(ValueError):
        FactorizedLinear(in_features=sizes[0], out_features=sizes[1], latent_features=sizes[2])


@pytest.mark.parametrize('sizes', [(8, 8, 3, 4), (7, 5, 2, 4), (12, 13, 0.3, 5)])
def test_factorized_conv2d_set_rank_throws_if_latent_too_big(sizes):
    module = FactorizedConv2d(in_channels=sizes[0], out_channels=sizes[1], kernel_size=1, latent_channels=sizes[2])
    X = torch.randn(2, module.in_channels, 8, 8)
    with pytest.raises(ValueError):
        module.set_rank(X, sizes[3])


@pytest.mark.parametrize('sizes', [(8, 8, 3, 4), (7, 5, 2, 4), (12, 13, 0.3, 5)])
def test_factorized_linear_set_rank_throws_if_latent_too_big(sizes):
    module = FactorizedLinear(in_features=sizes[0], out_features=sizes[1], latent_features=sizes[2])
    X = torch.randn(2, module.in_features)
    with pytest.raises(ValueError):
        module.set_rank(X, sizes[3])
