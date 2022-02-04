# Copyright 2021 MosaicML. All Rights Reserved.

import torch
from torchvision.models.resnet import Bottleneck


def _sample_drop(x: torch.Tensor, sample_drop_rate: float, is_training: bool):
    """Randomly drops samples from the input batch according to the `sample_drop_rate`.

    This is implemented by setting the samples to be dropped to zeros.
    """

    keep_probability = (1 - sample_drop_rate)
    if not is_training:
        return x * keep_probability
    rand_dim = [x.shape[0]] + [1] * len(x.shape[1:])
    sample_mask = keep_probability + torch.rand(rand_dim, dtype=x.dtype, device=x.device)
    sample_mask.floor_()  # binarize
    x *= sample_mask
    return x


class SampleStochasticBottleneck(Bottleneck):
    """Sample-wise stochastic ResNet Bottleneck block.

    This block has a probability of dropping samples before the identity
    connection, then adds back the untransformed samples using the identity
    connection.

    Args:
        sample_drop_rate (float): The probability of dropping a sample within
            this block. Must be between 0.0 and 1.0.
        **kwargs: Used for the original Bottleneck initialization arguments.
    """

    def __init__(self, drop_rate: float, **kwargs):
        super(SampleStochasticBottleneck, self).__init__(**kwargs)

        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.drop_rate:
            out = _sample_drop(out, self.drop_rate, self.training)
        out += identity

        return self.relu(out)

    @staticmethod
    def from_target_layer(module: Bottleneck, module_index: int, module_count: int, drop_rate: float,
                          drop_distribution: str):
        """Helper function to convert a ResNet Bottleneck block into a sample-wise stochastic block."""

        if drop_distribution == 'linear':
            drop_rate = ((module_index + 1) / module_count) * drop_rate

        return SampleStochasticBottleneck(drop_rate=drop_rate,
                                          inplanes=module.conv1.in_channels,
                                          planes=module.conv3.out_channels // module.expansion,
                                          stride=module.stride,
                                          downsample=module.downsample,
                                          groups=module.conv2.groups,
                                          dilation=module.conv2.dilation)
