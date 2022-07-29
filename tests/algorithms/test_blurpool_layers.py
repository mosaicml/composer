# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

from composer.algorithms import blurpool


def generate_pool_args():
    n_vals = [2]
    c_vals = [2]
    size_vals = [(3, 3), (3, 7), (4, 4)]
    strides = [1, 2]
    filter_size_vals = [(1, 1), (1, 3), (3, 3)]
    return list(itertools.product(n_vals, c_vals, size_vals, strides, filter_size_vals))


@pytest.mark.parametrize('pool_args', generate_pool_args())
def test_blurmaxpool_shapes(pool_args):
    n, c, sz, stride, kernel_size = pool_args

    X = torch.randn(n, c, sz[0], sz[1])

    layer_args = {'kernel_size': kernel_size, 'stride': stride, 'dilation': 1}
    blurpool_layer = blurpool.BlurMaxPool2d(**layer_args)
    maxpool_layer = torch.nn.MaxPool2d(**layer_args)

    assert blurpool_layer(X).shape == maxpool_layer(X).shape


@pytest.mark.parametrize('blur_first', [True, False])
@pytest.mark.parametrize('pool_args', generate_pool_args())
def test_blurconv2d_shapes(pool_args, blur_first):
    n, c, sz, stride, kernel_size = pool_args

    X = torch.randn(n, c, sz[0], sz[1])

    layer_args = {'kernel_size': kernel_size, 'stride': stride, 'dilation': 1, 'in_channels': c, 'out_channels': c + 1}
    blurconv2d_layer = blurpool.BlurConv2d(**layer_args, blur_first=blur_first)
    conv2d_layer = torch.nn.Conv2d(**layer_args)

    assert blurconv2d_layer(X).shape == conv2d_layer(X).shape


@pytest.mark.parametrize('pool_args', generate_pool_args())
def test_blur2d_shapes(pool_args):
    n, c, sz, _, _ = pool_args

    X = torch.randn(n, c, sz[0], sz[1])
    out = blurpool.blur_2d(X)
    assert out.shape == X.shape


def test_default_2d_filter():

    def reference_filter():
        filt = torch.FloatTensor([1, 2, 1])
        filt = torch.outer(filt, filt)
        filt *= 1. / filt.sum()
        filt = torch.Tensor(filt)
        return filt.view(1, 1, *filt.shape)

    torch.testing.assert_close(
        blurpool.blurpool_layers._default_2d_filter(),  # type: ignore
        reference_filter(),
    )


@pytest.mark.parametrize('pool_args', generate_pool_args())
def test_blur2d_std(pool_args):
    n, c, sz, _, _ = pool_args

    X = torch.randn(n, c, sz[0], sz[1])
    out = blurpool.blur_2d(X)
    assert torch.std(out) <= torch.std(X)


def test_blurpool_blurconv2d_params_match_original_params():
    conv2d = torch.nn.Conv2d(16, 32, 3, stride=1, bias=True)
    blurconv = blurpool.BlurConv2d.from_conv2d(conv2d)
    torch.testing.assert_close(blurconv.conv.weight, conv2d.weight)
    torch.testing.assert_close(blurconv.conv.bias, conv2d.bias)
    assert blurconv.conv.weight.requires_grad
    assert blurconv.conv.bias is not None
    assert blurconv.conv.bias.requires_grad
