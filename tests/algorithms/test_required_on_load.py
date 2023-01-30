# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import os
import pathlib
from typing import Type

import pytest
import torch

from composer import Trainer, algorithms
from composer.callbacks import CheckpointSaver
from composer.core import Algorithm, Time, TimeUnit  # type: ignore imports used in `eval(representation)`
from composer.models import ComposerClassifier, ComposerModel, composer_resnet
from tests.common import ConvModel


def initialize_algorithm(algo_cls: Type):
    """Initialize algorithm with dummy values."""
    if algo_cls == algorithms.Alibi:
        return algo_cls(max_sequence_length=1)
    elif algo_cls == algorithms.StochasticDepth:
        return algo_cls(target_layer_name='ResNetBottleneck')
    elif algo_cls == algorithms.FusedLayerNorm or algorithms.LowPrecisionLayerNorm:
        pytest.importorskip('apex')
        return algo_cls()
    elif algo_cls == algorithms.GatedLinearUnits:
        pytest.importorskip('transformers')
        return algo_cls()
    elif algo_cls == algorithms.Factorize:
        return algo_cls(min_features=48, latent_features=24)
    elif algo_cls == algorithms.SqueezeExcite:
        return algo_cls(min_channels=32)
    else:
        return algo_cls()


@pytest.mark.parametrize('algo_name', algorithms.__all__)
def test_required_on_load_has_repr(algo_name: str):
    algo_cls = getattr(algorithms, algo_name)
    if issubclass(algo_cls, Algorithm) and algo_cls.required_on_load():
        representation = repr(initialize_algorithm(algo_cls))
        # Default repr prints memory address
        assert 'at 0x' not in representation
        eval(f'algorithms.{representation}')


def compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module, is_equal: bool = True):
    """Check if two models are the same.

    To check equivalence, we first verify the modules are the same in each model by checking the
    list of attributes for each module. This checks if the same set of modules are used and if any
    module has been marked by adding an attribute. Next, we directly iterate over the state dict
    and verify the parameters are equal.

    This is not a comprehensive comparison. For example, an algorithm could silently monkeypatch
    over a forward function for a module. However, it is sufficient to provide coverage for our
    existing set of algorithms.
    """
    with contextlib.nullcontext() if is_equal else pytest.raises(Exception):
        # Compare model module attributes since algorithms like StochasticDepth monkeypatch
        # on new attributes. We only check this on ComposerClassifier models that have .module
        if isinstance(model_1, ComposerClassifier) and isinstance(model_2, ComposerClassifier):
            model_1_modules = list(model_1.module.modules())
            model_2_modules = list(model_2.module.modules())
            assert len(model_1_modules) == len(model_2_modules)
            for module_1, module_2 in zip(model_1_modules, model_2_modules):
                assert sorted(list(module_1.__dict__.keys())) == sorted(list(module_2.__dict__.keys()))
        # Compare model parameters
        for (name0, tensor0), (name1, tensor1) in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            assert name0 == name1
            assert torch.equal(tensor0, tensor1)


@pytest.mark.filterwarnings('ignore:No instances of')
@pytest.mark.parametrize('algo_name', algorithms.__all__)
def test_idempotent(algo_name: str, tiny_bert_config):
    algo_cls = getattr(algorithms, algo_name)
    if issubclass(algo_cls, Algorithm) and algo_cls.required_on_load():
        algorithm = initialize_algorithm(algo_cls)

        original_model = None
        if algo_name == 'StochasticDepth':
            original_model = composer_resnet(model_name='resnet50')
        elif algo_name in ['Alibi', 'GatedLinearUnits']:
            transformers = pytest.importorskip('transformers')
            from composer.models import HuggingFaceModel
            hf_model = transformers.AutoModelForSequenceClassification.from_config(tiny_bert_config)
            original_model = HuggingFaceModel(hf_model, use_logits=True)
        else:
            original_model = ConvModel()
        applied_once_model = Trainer(
            model=copy.deepcopy(original_model),
            algorithms=algorithm,
        ).state.model
        assert isinstance(applied_once_model, ComposerModel)  # Assert type for pyright deepcopy
        applied_twice_model = Trainer(
            model=copy.deepcopy(applied_once_model),
            algorithms=algorithm,
        ).state.model
        compare_models(original_model, applied_twice_model, is_equal=False)  # Surgery actually changes model
        compare_models(applied_once_model, applied_twice_model, is_equal=True)  # Multiple applications are no-ops


@pytest.mark.parametrize('algo_name', algorithms.__all__)
@pytest.mark.parametrize('load_weights_only,already_added,exclude', [
    [False, False, False],
    [True, False, False],
    [False, True, False],
    [False, False, True],
])
def test_autoload(algo_name: str, load_weights_only: bool, already_added: bool, exclude: bool, tmp_path: pathlib.Path,
                  tiny_bert_config):
    algo_cls = getattr(algorithms, algo_name)
    if issubclass(algo_cls, Algorithm) and algo_cls.required_on_load():
        algorithm = initialize_algorithm(algo_cls)

        original_model = None
        if algo_name == 'StochasticDepth':
            original_model = composer_resnet(model_name='resnet50')
        elif algo_name in ['Alibi', 'GatedLinearUnits']:
            transformers = pytest.importorskip('transformers')
            from composer.models import HuggingFaceModel
            hf_model = transformers.AutoModelForSequenceClassification.from_config(tiny_bert_config)
            original_model = HuggingFaceModel(hf_model, use_logits=True)
        else:
            original_model = ConvModel()

        trainer1 = Trainer(model=copy.deepcopy(original_model),
                           algorithms=algorithm,
                           save_folder=str(tmp_path),
                           save_filename='ckpt.pt')
        checkpoint_saver = [cb for cb in trainer1.state.callbacks if isinstance(cb, CheckpointSaver)][0]
        checkpoint_saver._save_checkpoint(trainer1.state, trainer1.logger)

        context = contextlib.nullcontext()
        # Emit warning when autoloading
        if not already_added and not exclude:
            context = pytest.warns(UserWarning, match='Automatically adding required_on_load algorithm*')
        # Excluding some algorithms leads to errors when loading
        elif exclude:
            if algo_name in ['Factorize', 'SqueezeExcite']:
                context = pytest.raises(
                    ValueError,
                    match=
                    "loaded state dict contains a parameter group that doesn't match the size of optimizer's group",
                )
            elif algo_name == 'Alibi':
                context = pytest.raises(RuntimeError)

        with context:
            trainer2 = Trainer(
                model=copy.deepcopy(original_model),
                algorithms=[initialize_algorithm(algo_cls)] if already_added else [],
                load_path=os.path.join(str(tmp_path), 'ckpt.pt'),
                load_weights_only=load_weights_only,
                load_exclude_algorithms=[algo_name] if exclude else None,
            )
            # No algorithms are added if we've excluded them
            if exclude:
                assert len(trainer2.state.algorithms) == 0
            # Otherwise, check exactly one copy of algorithm is present
            else:
                assert len(trainer2.state.algorithms) == 1
                assert isinstance(trainer2.state.algorithms[0], algo_cls)
