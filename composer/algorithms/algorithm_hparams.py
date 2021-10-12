from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import yahp as hp

import composer
from composer.core.algorithm import Algorithm


@dataclass
class AlgorithmHparams(hp.Hparams, ABC):

    @abstractmethod
    def initialize_object(self) -> Algorithm:
        pass

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    @classmethod
    def load(cls) -> AlgorithmHparams:
        from composer.algorithms.algorithm_registry import get_algorithm_registry
        registry = get_algorithm_registry()
        inverted_registry = {v: k for (k, v) in registry.items()}
        alg_name = inverted_registry[cls]
        hparams_file = os.path.join(os.path.dirname(composer.__file__), "yamls", "algorithms", alg_name, "hparams.yaml")
        if os.path.exists(hparams_file):
            alg_hparams = cls.create(hparams_file)
            assert isinstance(alg_hparams, AlgorithmHparams), "hparams.create should return an instance of its type"
            return alg_hparams
        return cls()

    @classmethod
    def load_multiple(cls, *algorithms: str):
        from composer.algorithms.algorithm_registry import get_algorithm_registry
        registry = get_algorithm_registry()
        try:
            return [registry[alg].load() for alg in algorithms]
        except KeyError as e:
            raise ValueError(f"Algorithm {e.args[0]} not found") from e
