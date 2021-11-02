# Copyright 2021 MosaicML. All Rights Reserved.

import torch
from torch import nn
from torch.nn.common_types import _size_2_t

from composer.algorithms.factorize.factorize_core import (FractionOrInt, LowRankSolution,
                                                          apply_solution_to_module_parameters, clean_latent_dims,
                                                          factorize_conv2d, max_rank_with_possible_speedup)


class FactorizedConv2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 latent_channels: FractionOrInt = .5,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.latent_channels = clean_latent_dims(latent_channels, self.in_channels, self.out_channels)

        # conv2d factorization code requires most Conv2d arguments, but
        # not boolean 'bias'
        self.convolution_kwargs = {k: v for k, v in kwargs.items() if k != 'bias'}

        self.already_factorized = False
        if self._should_factorize():
            # this one produces identical output as a regular Conv2d would,
            # except with fewer output channels
            self.conv0 = nn.Conv2d(self.in_channels, self.latent_channels, self.kernel_size, bias=False, **self.convolution_kwargs)
            # this one increases the number of output channels
            self.conv1 = nn.Conv2d(self.latent_channels, self.out_channels, kernel_size=1, bias=True)
            self.already_factorized = True
        else:
            self.latent_channels = self.out_channels
            self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, **kwargs)
            self.conv1 = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = self.conv0(input)
        if self.conv1 is not None:
            ret = self.conv1(ret)
        return ret

    def _should_factorize(self):
        if self.already_factorized:
            max_rank = self.latent_channels - 1
        else:
            max_rank = max_rank_with_possible_speedup(self.in_channels, self.out_channels, self.kernel_size)
        return self.latent_channels <= max_rank

    def _solution_for_rank(self, input: torch.Tensor, rank: int) -> LowRankSolution:
        weight0 = self.conv0.weight
        bias0 = self.conv0.bias
        if self.conv1 is None:
            weight1, bias1 = None, None
        else:
            weight1, bias1 = self.conv1.weight, self.conv1.bias

        return factorize_conv2d(input,
                                weight0,
                                weight1,
                                rank=rank,
                                biasA=bias0,
                                biasB=bias1,
                                **self.convolution_kwargs)

    def reset_parameters(self):
        self.conv0.reset_parameters()
        if self.conv1 is not None:
            self.conv1.reset_parameters()

    def set_rank(self, input: torch.Tensor, rank: int) -> None:
        soln = self._solution_for_rank(input, rank)
        self._update_factorization(soln)

    def _update_factorization(self, solution: LowRankSolution):
        self.latent_channels = solution.rank
        if not self.already_factorized:
            self.conv1 = nn.Conv2d(self.latent_channels, self.out_channels, kernel_size=1, bias=True)
            self.already_factorized = True
        self.conv0.out_channels = solution.rank
        self.conv1.in_channels = solution.rank
        apply_solution_to_module_parameters(solution, self.conv0, self.conv1, transpose=False)

    @staticmethod
    def from_conv2d(module: torch.nn.Conv2d, module_ix: int = -1, **kwargs):
        conv = FactorizedConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=((module.bias is not None) and (module.bias is not False)),
            **kwargs  # custom params
        )
        conv.reset_parameters()
        return conv
