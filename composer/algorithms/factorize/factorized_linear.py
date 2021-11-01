# Copyright 2021 MosaicML. All Rights Reserved.

import torch
from torch import nn

from composer.algorithms.factorize.factorize_core import (FractionOrInt, LowRankSolution,
                                                          apply_solution_to_module_parameters, clean_latent_dims,
                                                          factorize, max_rank_with_possible_speedup)


class FactorizedLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, latent_features: FractionOrInt = .5, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.latent_features = clean_latent_dims(latent_features, self.in_features, self.out_features)
        self.bias = bias

        self.already_factorized = False
        if self._should_factorize():
            self.linear0 = nn.Linear(in_features=self.in_features, out_features=self.latent_features, bias=False)
            self.linear1 = nn.Linear(in_features=self.latent_features, out_features=self.out_features, bias=self.bias)
        else:
            self.latent_features = min(self.in_features, self.out_features)
            self.linear0 = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
            self.linear1 = None

    def reset_parameters(self):
        self.linear0.reset_parameters()
        if self.linear1 is not None:
            self.linear1.reset_parameters()

    def set_rank(self, input: torch.Tensor, rank: int) -> None:
        soln = self._solution_for_rank(input, rank)
        self._update_factorization(soln)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = self.linear0(input)
        if self.linear1 is not None:
            ret = self.linear1(ret)
        return ret

    def _should_factorize(self):
        if self.already_factorized:
            max_rank = self.latent_features - 1
        else:
            max_rank = max_rank_with_possible_speedup(self.in_features, self.out_features)
        return self.latent_features <= max_rank

    def _solution_for_rank(self, input: torch.Tensor, rank: int) -> LowRankSolution:
        weight0 = self.linear0.weight
        bias0 = self.linear0.bias
        if self.linear1 is None:
            weight1, bias1 = None, None
        else:
            weight1, bias1 = self.linear1.weight, self.linear1.bias

        return factorize(input,
                         weight0,
                         weight1,
                         rank=rank,
                         biasA=bias0,
                         biasB=bias1)

    def _update_factorization(self, solution: LowRankSolution):
        self.latent_features = solution.rank
        if not self.already_factorized:
            self.linear1 = nn.Linear(self.latent_features, self.out_features, bias=True)
            self.already_factorized = True
        self.linear0.out_features = solution.rank
        self.linear1.in_features = solution.rank
        apply_solution_to_module_parameters(solution, self.linear0, self.linear1)

    @staticmethod
    def from_linear(module: torch.nn.Linear, module_ix: int = -1, **kwargs):
        ret = FactorizedLinear(in_features=module.in_features,
                               out_features=module.out_features,
                               bias=((module.bias is not None) and (module.bias is not False)),
                               **kwargs)
        ret.reset_parameters()
        return ret
