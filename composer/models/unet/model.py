# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Unet architecture used in image segmentation. The example we are using is for BRATS medical brain tumor dataset.

See the :doc:`Model Card </model_cards/unet>` for more details.
"""

import torch.nn as nn

from composer.models.unet._layers import ConvBlock, OutputBlock, ResidBlock, UpsampleBlock

__all__ = ['UNet']


class UNet(nn.Module):
    """Unet Architecture adapted from NVidia `Deep Learning Examples`_.

    .. _Deep Learning Examples: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/nnUNet/

    Args:
        in_channels (int): Number of input channels.
        n_class (int): Number of output layers.
        kernels (list): Conv layer kernel sizes.
        strides (list): Conv layer strides.
        normalization_layer (str): Normalization layer type, one of (``"batch"``, ``"instance"``).
        negative_slope (float): Leaky relu negative slope.
        residual (bool): Use residual connections.
        dimension (int): Filter dimensions.
    """

    def __init__(
        self,
        in_channels,
        n_class,
        kernels,
        strides,
        normalization_layer,
        negative_slope,
        residual,
        dimension,
    ):
        super(UNet, self).__init__()
        self.dim = dimension
        self.n_class = n_class
        self.residual = residual
        self.negative_slope = negative_slope
        self.norm = normalization_layer + f'norm{dimension}d'
        self.filters = [min(2**(5 + i), 320 if dimension == 3 else 512) for i in range(len(strides))]

        down_block = ResidBlock if self.residual else ConvBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            in_channels=in_channels,
            out_channels=self.filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
        )
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
        )
        self.upsamples = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[1:][::-1],
            out_channels=self.filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block(decoder_level=0)
        self.apply(self.initialize_weights)
        self.n_layers = len(self.upsamples) - 1

    def forward(self, input_data):
        out = self.input_block(input_data)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        out = self.bottleneck(out)
        for idx, upsample in enumerate(self.upsamples):
            out = upsample(out, encoder_outputs[self.n_layers - idx])
        out = self.output_block(out)
        return out

    def get_conv_block(self, conv_block, in_channels, out_channels, kernel_size, stride):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            negative_slope=self.negative_slope,
        )

    def get_output_block(self, decoder_level):
        return OutputBlock(in_channels=self.filters[decoder_level], out_channels=self.n_class, dim=self.dim)

    def get_module_list(self, in_channels, out_channels, kernels, strides, conv_block):
        layers = []
        for in_channel, out_channel, kernel, stride in zip(in_channels, out_channels, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, in_channel, out_channel, kernel, stride)
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ['conv2d']:
            nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
