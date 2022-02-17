# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import yahp as hp

import composer
from composer.core.algorithm import Algorithm


@dataclass
class AlgorithmHparams(hp.Hparams, ABC):
    """Hyperparameters for algorithms."""

    @abstractmethod
    def initialize_object(self) -> Algorithm:
        """Invoked by the :meth:`TrainerHparams.initialize_object` to create an instance of the :class:`Algorithm`.

        Returns:
            Algorithm: An instance of the :class:`Algorithm`.
        """
        pass

    @classmethod
    def load(cls, alg_params: Optional[str] = None) -> AlgorithmHparams:
        from composer.algorithms.algorithm_registry import get_algorithm_registry
        registry = get_algorithm_registry()
        inverted_registry = {v: k for (k, v) in registry.items()}
        alg_name = inverted_registry[cls]
        alg_folder = os.path.join(os.path.dirname(composer.__file__), "yamls", "algorithms")
        if alg_params is None:
            hparams_file = os.path.join(alg_folder, f"{alg_name}.yaml")
        else:
            hparams_file = os.path.join(alg_folder, alg_name, f"{alg_params}.yaml")
        if os.path.exists(hparams_file):
            alg_hparams = cls.create(hparams_file, cli_args=False)
            assert isinstance(alg_hparams, AlgorithmHparams), "hparams.create should return an instance of its type"
            return alg_hparams
        return cls()

    @classmethod
    def load_multiple(cls, *algorithms: str):
        from composer.algorithms.algorithm_registry import get_algorithm_registry
        registry = get_algorithm_registry()
        alg_hparams = []
        for alg in algorithms:
            alg_parts = alg.split("/")
            alg_name = alg_parts[0]
            if len(alg_parts) > 1:
                alg_params = "/".join(alg_parts[1:])
            else:
                alg_params = None
            try:
                alg_param = registry[alg_name]
            except KeyError as e:
                raise ValueError(f"Algorithm {e.args[0]} not found") from e
            alg_hparams.append(alg_param.load(alg_params))
        return alg_hparams
