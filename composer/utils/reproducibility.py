# Copyright 2021 MosaicML. All Rights Reserved.

"""Helper utilities for configuring deterministic training and ensuring reproducibility."""
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn

__all__ = [
    "configure_deterministic_mode",
    "get_random_seed",
    "seed_all",
]


def configure_deterministic_mode():
    """Configure PyTorch deterministic mode.

    .. note::

        When using the :class:`~composer.trainer.trainer.Trainer`, use the ``determinstic_mode`` flag
        instead of invoking this function directly.
        For example:

        >>> trainer = Trainer(determinstic_mode=True)
        trainer

        This is provided for convenience if deterministic operations must be performed before
        initializing the trainer.

    .. note::
        
        When training on a GPU, :meth:`configure_deterministic_mode` must be invoked before any CUDA operations.

    .. note::

        Deterministic mode degrades performance. Its use should be limited to testing and debugging.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    # and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    warnings.warn("Deterministic mode is activated. This will negatively impact performance.", category=UserWarning)


def get_random_seed() -> int:
    """Get a randomly created seed to use for seeding rng objects.

    .. warning::

        This random seed is NOT cryptographically secure.

    Returns:
        int: A random seed.
    """
    seed = int(torch.empty((), dtype=torch.int64).random_(to=2**32).item())
    return seed


def seed_all(seed: int):
    """Seed all rng objects.

    Args:
        seed (int): The random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)
