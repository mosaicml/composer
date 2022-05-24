import torch
from torch import nn
import numpy as np
import torch.nn.utils.parametrize as parametrize
from composer.core import Algorithm

__all__ = ["apply_weight_standardization", "WeightStandardization"]

def _standardize_weights(W: torch.Tensor):
    reduce_dims = list(range(1, W.dim()))
    return (W - W.mean(dim=reduce_dims, keepdim=True)) / W.std(dim=reduce_dims, keepdim=True)
    

class WeightStandardizer(nn.Module):
    def forward(self, W):
        return _standardize_weights(W)
        

def apply_weight_standardization(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parametrize.register_parametrization(module, "weight", WeightStandardizer())

class WeightStandardization(Algorithm):
    pass