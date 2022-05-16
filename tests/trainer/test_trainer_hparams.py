from __future__ import annotations

import types
import typing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Type

import pytest

import composer.algorithms
import composer.callbacks
from composer.algorithms.algorithm_hparams import algorithm_registry
from composer.callbacks import Callback
from composer.callbacks.callback_hparams import callback_registry
from composer.core import Algorithm
from composer.datasets.dataloader import DataLoaderHparams
from composer.trainer import EvalHparams, ExperimentHparams, FitHparams, Trainer, TrainerHparams
from composer.trainer.trainer_hparams import EvalKwargs, FitKwargs
from tests.common import SimpleModelHparams
from tests.common.datasets import RandomClassificationDatasetHparams

if TYPE_CHECKING:
    from typing import TypedDict


@pytest.mark.parametrize("method,typeddict_cls", [[Trainer.fit, FitKwargs], [Trainer.eval, EvalKwargs]])
def test_kwargs_match_signature(method: Callable, typeddict_cls: Type[TypedDict]):
    assert typing.get_type_hints(method) == typing.get_type_hints(typeddict_cls)


def test_experiment_hparams_initialize():
    """Test that the ExperimentHparams class initializes."""
    experiment_hparams = ExperimentHparams(
        trainer=TrainerHparams(
            model=SimpleModelHparams(),
            dataloader=DataLoaderHparams(
                num_workers=0,
                persistent_workers=False,
                pin_memory=False,
            ),
            max_duration=1,
        ),
        fits=[
            FitHparams(
                train_dataset=RandomClassificationDatasetHparams(),
                train_batch_size=1,
                train_subset_num_batches=1,
            ),
        ],
        evals=[EvalHparams(
            dataset=RandomClassificationDatasetHparams(),
            batch_size=1,
            subset_num_batches=1,
        )])
    trainer, fits, evals = experiment_hparams.initialize_object()

    for fit_kwargs in fits:
        trainer.fit(**fit_kwargs)

    for eval_kwargs in evals:
        trainer.eval(**eval_kwargs)


@pytest.mark.parametrize(
    "module,registry,cls,ignore_list",
    [
        [composer.callbacks, callback_registry, Callback, []],
        [composer.algorithms, algorithm_registry, Algorithm, []],
        # TODO(ravi) -- add in other classes as they are de-yahpified
    ])
def test_registry_contains_all_entries(module: types.ModuleType, registry: Dict[str, Callable], cls: Type,
                                       ignore_list: List[Callable]):
    # for each module, extract the items from `__all__` that of type `cls`
    # then, assert that the registry contains an entry for each of type `cls`
    # skip any entries in `ignore_list` since they may be auto-initialized hparams
    subclasses = [x for x in vars(module).values() if isinstance(x, type) and issubclass(x, cls)]
    registry_entries = set(registry.values())
    registry_entries.union(ignore_list)
    for subclass in subclasses:
        assert subclass in registry_entries, f"Class {type(cls).__name__} is missing from the registry."


@pytest.mark.parametrize("registry,defaults", [
    [callback_registry, {}],
    [algorithm_registry, {}],
])
def test_registry_initializes(registry: Dict[str, Callable], defaults: Dict[str, Dict[str, Any]]):
    for name, constructor in registry.items():
        constructor_defaults = defaults.get(name, {})
        constructor(**constructor_defaults)
