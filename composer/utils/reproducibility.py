# Copyright 2021 MosaicML. All Rights Reserved.

"""Helper utilities for configuring deterministic training and ensuring reproducibility.

.. note::

    For a deterministic model initialization, :meth:`~composer.utils.reproducibility.seed_all` and/or
    :meth:`~composer.utils.configure_deterministic_mode` should be
    invoked before creating and initializing a model, before creating the :class:`~composer.trainer.trainer.Trainer`.
    For example:

    .. testsetup::
        import torch.nn

        class MyModel(torch.nn.Module):
            pass

    .. doctest::

        >>> import torch.nn.init
        >>> from composer.utils import reproducibility
        >>> reproducibility.configure_deterministic_mode()
        >>> reproducibility.seed_all(42)
        >>> model = MyModel()
        >>> # model will now be deterministically initialized, since the seed is set.
        >>> model.apply(torch.nn.init.kaiming_uniform_)
        >>> trainer = Trainer(model)
"""
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

        When using the :class:`~composer.trainer.trainer.Trainer`, you can use the ``deterministic_mode`` flag
        instead of invoking this function directly.
        For example:

        >>> trainer = Trainer(deterministic_mode=True)

        However, if deterministic operations must be performed before initializing the trainer
        (such as model initialization), then this function can be called beforehand.

    .. note::
        
        When training on a GPU, this function must be invoked before any CUDA operations.

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

    .. note::

        When using the :class:`~composer.trainer.trainer.Trainer`, you can use the ``seed`` parameter
        instead of invoking this function directly.
        For example:

        >>> trainer = Trainer(seed=42)

        However, if deterministic operations must be performed before initializing the trainer
        (such as model initialization), then this function can be called beforehand.

    Args:
        seed (int): The random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)
