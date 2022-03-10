# Copyright 2021 MosaicML. All Rights Reserved.

"""Helper utilities for configuring deterministic training to ensure reproducibility.

.. note::

    For deterministic model initialization, :meth:`~composer.utils.reproducibility.seed_all` and/or
    :meth:`~composer.utils.configure_deterministic_mode` should be
    invoked before creating and initializing a model, before creating the :class:`~composer.trainer.trainer.Trainer`.
    For example:

    .. testsetup::

        import functools
        import torch.nn

        MyModel = functools.partial(SimpleBatchPairModel, num_channels, num_classes)

    .. doctest::

        >>> import torch.nn
        >>> from composer.utils import reproducibility
        >>> reproducibility.configure_deterministic_mode()
        >>> reproducibility.seed_all(42)
        >>> model = MyModel()
        >>> def init_weights(m):
        ...     if isinstance(m, torch.nn.Linear):
        ...         torch.nn.init.xavier_uniform(m.weight)
        >>> # model will now be deterministically initialized, since the seed is set.
        >>> init_weights(model)
        >>> trainer = Trainer(model=model)

Attributes:
    MAX_SEED (int): The maximum allowed seed, which is :math:`2^{32} - 1`.
"""

from __future__ import annotations

import os
import random
import textwrap
import time
import warnings
from typing import TYPE_CHECKING, List

import numpy as np
import torch
import torch.backends.cudnn

from composer.utils import dist

if TYPE_CHECKING:
    from composer.core import types

__all__ = [
    "configure_deterministic_mode",
    "get_random_seed",
    "seed_all",
    "get_rng_state",
    "load_rng_state",
    "MAX_SEED",
]

# seeds must be 32-bit unsigned integers
MAX_SEED = 2**32 - 1


def configure_deterministic_mode():
    """Configure PyTorch deterministic mode.

    .. note::

        When using the :class:`~composer.trainer.trainer.Trainer`, you can use the ``deterministic_mode`` flag
        instead of invoking this function directly.
        For example:

        >>> trainer = Trainer(deterministic_mode=True)

        However, to configure deterministic mode for operations before the trainer is initialized, manually invoke this
        function at the beginning of your training script.

    .. note::
        
        When training on a GPU, this function must be invoked before any CUDA operations.

    .. note::

        Deterministic mode degrades performance. Do not use outside of testing and debugging.
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
    rng = random.Random(int(time.time_ns()))  # get a new RNG does not respect the current seed
    seed = rng.randint(0, MAX_SEED)
    assert seed >= 0 and seed <= MAX_SEED, "seed should be on this range"
    return seed


def seed_all(seed: int):
    """Seed all rng objects.

    .. note::

        When using the :class:`~composer.trainer.trainer.Trainer`, you can use the ``seed`` parameter
        instead of invoking this function directly.
        For example:

        >>> trainer = Trainer(seed=42)

        However, to configure the random seed for operations before the trainer is initialized, manually invoke this
        function at the beginning of your training script.

    Args:
        seed (int): The random seed
    """
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> List[types.StateDict]:
    """The state of the RNG objects.

    Returns:
        List[types.StateDict]: A list of RNG State Dicts, indexed by global rank.
    """

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # This will not be compatible with model parallelism
        rng_state['cuda'] = torch.cuda.get_rng_state()

    return dist.all_gather_object(rng_state)


def load_rng_state(rng_state_dicts: List[types.StateDict]):
    """Restore the RNG state.

    Args:
        rng_state_dicts (List[types.StateDict]): The list of RNG state dicts to restore,
            as returned by :func:`get_rng_state`.
    """
    if dist.get_world_size() > len(rng_state_dicts):
        warnings.warn(
            textwrap.dedent(f"""\
                The current world size ({dist.get_world_size()} is greater than the number of RNG state(s) serialized
                ({len(rng_state_dicts)}). Only the first {len(rng_state_dicts)} rank(s) will have their RNG restored.
                """))
    if dist.get_world_size() < len(rng_state_dicts):
        warnings.warn(
            textwrap.dedent(f"""\
            The current world size ({dist.get_world_size()} is less than the number of RNG state(s) serialized
            ({len(rng_state_dicts)}). Only the first {dist.get_world_size()} RNG state(s) will be consumed;
            the remaining will be ignored."""))

    if dist.get_global_rank() < len(rng_state_dicts):
        rng_state_dict = rng_state_dicts[dist.get_global_rank()]
        torch.set_rng_state(rng_state_dict['torch'])
        random.setstate(rng_state_dict['python'])
        np.random.set_state(rng_state_dict['numpy'])

        is_cuda_available = torch.cuda.is_available() and torch.cuda.is_initialized()
        has_cuda_rng_state = "cuda" in rng_state_dict

        if is_cuda_available and has_cuda_rng_state:
            torch.cuda.set_rng_state(rng_state_dict['cuda'])

        if is_cuda_available and not has_cuda_rng_state:
            warnings.warn(
                textwrap.dedent(f"""\
                The checkpoint did not include the CUDA RNG state. The CUDA RNG will have a
                non-deterministic state."""))
        if not is_cuda_available and has_cuda_rng_state:
            warnings.warn(
                textwrap.dedent(f"""\
                The checkpoint included CUDA RNG state, but CUDA is not being used.
                As such, the CUDA RNG state will be ignored."""))
