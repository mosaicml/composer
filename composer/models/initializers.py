# Copyright 2021 MosaicML. All Rights Reserved.

import torch
import torch.nn as nn


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
        w.weight.data = torch.ones_like(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)


def bn_uniform(w: nn.Module):
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)


def get_initializer(name: str):
    initializer_dict = {
        "kaiming_normal": kaiming_normal,
        "kaiming_uniform": kaiming_uniform,
        "bn_uniform": bn_uniform,
        "bn_ones": bn_ones,
        "xavier_uniform": xavier_uniform
    }
    if name not in initializer_dict:
        raise ValueError(f"Initializer '{name}' not found.")
    return initializer_dict[name]
