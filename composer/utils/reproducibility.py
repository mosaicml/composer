# Copyright 2021 MosaicML. All Rights Reserved.

import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn

_DETERMINISTIC_MODE_KEY = "COMPOSER_USE_DETERMINISTIC_MODE"


# def use_deterministic_mode():
#     return bool(int(os.environ.get(_DETERMINISTIC_MODE_KEY, "0")))


def configure_deterministic_mode():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    # and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    warnings.warn("Deterministic mode is activated. This will negatively impact performance.", category=UserWarning)


def get_random_seed() -> int:
    """Get a randomly created seed to use for seeding rng objects."""
    seed = int(torch.empty((), dtype=torch.int64).random_(to=2**32).item())
    return seed


def seed_all(seed: int):
    """Seed all rng objects

    Args:
        seed (int): random seed
        device (Optional[Device]): the
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)
