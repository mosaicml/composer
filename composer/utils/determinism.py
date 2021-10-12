# Copyright 2021 MosaicML. All Rights Reserved.

import random

import numpy as np
import torch


def get_random_seed() -> int:
    seed = int(torch.empty((), dtype=torch.int64).random_(to=2**32).item())
    return seed


def seed_all(seed: int):
    """
    Seed all rng objects

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
