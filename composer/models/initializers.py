from typing import Callable

import torch
from torch import nn as nn

from composer.utils import StringEnum


class Initializer(StringEnum):
    """Sets the initialization scheme for different layers of a PyTorch model."""
    KAIMING_NORMAL = "kaiming_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    BN_UNIFORM = "bn_uniform"
    BN_ONES = "bn_ones"
    XAVIER_UNIFORM = "xavier_uniform"

    def get_initializer(self) -> Callable[[torch.nn.Module], None]:

        def kaiming_normal(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(w.weight)

        def kaiming_uniform(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(w.weight)

        def xavier_uniform(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(w.weight)

        def bn_ones(w: nn.Module):
            if isinstance(w, torch.nn.BatchNorm2d):
                w.weight.data = torch.ones_like(w.weight.data)
                w.bias.data = torch.zeros_like(w.bias.data)

        def bn_uniform(w: nn.Module):
            if isinstance(w, torch.nn.BatchNorm2d):
                w.weight.data = torch.rand(w.weight.data.shape)
                w.bias.data = torch.zeros_like(w.bias.data)

        initializer_dict = {
            "kaiming_normal": kaiming_normal,
            "kaiming_uniform": kaiming_uniform,
            "bn_uniform": bn_uniform,
            "bn_ones": bn_ones,
            "xavier_uniform": xavier_uniform
        }
        if self.value not in initializer_dict:
            raise ValueError(f"Initializer '{self.value}' not found.")
        return initializer_dict[self.value]
