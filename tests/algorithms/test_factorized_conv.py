# Copyright 2021 MosaicML. All Rights Reserved.

import itertools

import numpy as np
import torch

from composer.algorithms.factorize import FactorizedConv2d


def _default_input_and_weight_shapes():
    batch_sizes = [1, 2]
    hs = [5]
    ws = [6]
    C_ins = [4, 8]
    C_outs = [4, 8]
    kernel_sizes = [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1), (5, 5)]
    return itertools.product(batch_sizes, hs, ws, C_ins, C_outs, kernel_sizes)


def _test_shapes(batch_size, h, w, C_in, C_out, kernel_size, **kwargs):
    X = torch.randn(batch_size, C_in, h, w)
    conv = FactorizedConv2d(in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, **kwargs)
    Y = conv(X)

    assert len(Y.shape) == 4
    assert np.allclose(Y.shape[:2], torch.Tensor([batch_size, C_out]))


def test_init_and_forward_run():
    for args in _default_input_and_weight_shapes():
        _test_shapes(*args)


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
