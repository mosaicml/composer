from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn.common_types import _size_2_t

from composer.algorithms import factorize

FractionOrInt = Union[int, float]


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
        self.latent_channels = self._clean_latent_channels(latent_channels)

        # conv2d factorization code requires most Conv2d arguments, but
        # not boolean 'bias'
        self.convolution_kwargs = {k: v for k, v in kwargs.items() if k != 'bias'}

        self.already_factorized = False
        max_rank = self._max_rank_with_possible_speedup()
        should_factorize = self.latent_channels <= max_rank

        if should_factorize:
            # this one produces identical output a regular Conv2d would, except
            # with fewer output channels
            self.conv0 = nn.Conv2d(self.in_channels, self.latent_channels, self.kernel_size, **kwargs)
            # this one increases the number of output channels
            self.conv1 = nn.Conv2d(self.latent_channels, self.out_channels, kernel_size=1)
            self.already_factorized = True
        else:
            self.latent_channels = min(self.in_channels, self.out_channels)
            self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, **kwargs)
            self.conv1 = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = self.conv0(input)
        if self.conv1 is not None:
            ret = self.conv1(ret)
        return ret

    def _max_rank_with_possible_speedup(self) -> int:
        if self.already_factorized:
            return self.latent_channels - 1
        # TODO less naive cost model than counting multiply adds
        fan_in = self.in_channels * np.prod(self.kernel_size)
        return (fan_in * self.out_channels) / (fan_in + self.out_channels) - 1

    def _clean_latent_channels(self, latent_channels: FractionOrInt) -> int:
        if latent_channels <= 1:  # fraction of input or output channels
            latent_channels = int(latent_channels * min(self.in_channels, self.out_channels))
            return max(1, latent_channels)
        return int(latent_channels)

    def _solution_for_rank(self, input: torch.Tensor, rank: int):
        weight0 = self.conv0.weight
        bias0 = self.conv0.bias
        if self.conv1 is None:
            weight1, bias1 = None, None
        else:
            weight1, bias1 = self.conv1.weight, self.conv1.bias

        return factorize.factorize_conv2d(input,
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

    def _update_factorization(self, solution: factorize.LowRankSolution):
        W0 = solution.Wa
        W1 = solution.Wb
        bias = solution.bias
        rank = solution.rank
        self.latent_channels = rank

        if not self.already_factorized:
            # below condition *should* never happen unless external algorithms
            # mess with our state, but it will produce confusing errors
            # if it does; fail fast and explicitly instead
            assert self.conv1 is None, "self.conv1 exists even though not self.already_factorized!"
            self.conv1 = nn.Conv2d(self.latent_channels, self.out_channels, kernel_size=1)
            self.already_factorized = True

        # mess with model's shape data
        self.conv0.out_channels = rank
        self.conv1.in_channels = rank
        with torch.no_grad():
            # first conv always has no bias since adds no expressivity
            self.conv0.bias = torch.nn.Parameter(torch.zeros(rank, dtype=self.conv0.bias.dtype))

            self.conv1.bias.copy_(bias)
            self.conv0.weight = torch.nn.Parameter(W0)
            self.conv1.weight = torch.nn.Parameter(W1)

    def speedup(self, input: torch.Tensor, max_nmse: Optional[float], rank_multiple_of: int = 32) -> bool:
        if max_nmse <= 0:
            return False
        if isinstance(input, (tuple, list)):
            # forward_pre_hooks get positional args passed in as a list, even
            # if there's only one of them
            input = input[0]

        max_rank = self._max_rank_with_possible_speedup()
        min_rank = rank_multiple_of

        if max_rank < min_rank:
            return  # no possible solution

        allowed_ranks = np.arange(min_rank // rank_multiple_of, max_rank // rank_multiple_of + 1) * rank_multiple_of

        # naive brute force search for simplicity; important to start from
        # lower ranks since these are much cheaper to evaluate
        # TODO binary or line search
        # TODO precompute reusable stats once to reuse across all lstsq() calls
        best_solution = None
        for rank in allowed_ranks:
            solution = self._solution_for_rank(input=input, rank=rank)
            if solution.nmse < max_nmse:
                best_solution = solution
                break

        if best_solution is None:
            return False  # no acceptable solution
        self._update_factorization(best_solution),
        return True

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
