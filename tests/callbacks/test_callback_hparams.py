# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type, Union

import pytest
import yahp as hp

from composer.callbacks.callback_hparams_registry import callback_registry
from composer.core import Callback
from composer.loggers import ObjectStoreLogger
from composer.loggers.logger_hparams_registry import ObjectStoreLoggerHparams, logger_registry
from composer.profiler import JSONTraceHandler, SystemProfiler, TorchProfiler, TraceHandler
from tests.callbacks.callback_settings import get_cb_hparams_and_marks, get_cb_kwargs, get_cbs_and_marks
from tests.common.hparams import assert_in_registry, construct_from_yaml


@pytest.mark.parametrize('constructor', get_cb_hparams_and_marks())
def test_callback_hparams_is_constructable(
    constructor: Union[Type[Callback], Type[hp.Hparams]],
    monkeypatch: pytest.MonkeyPatch,
):
    # The ObjectStoreLogger needs the KEY_ENVIRON set
    yaml_dict = get_cb_kwargs(constructor)
    if constructor is ObjectStoreLoggerHparams:
        monkeypatch.setenv('KEY_ENVIRON', '.')
    expected = ObjectStoreLoggerHparams if constructor is ObjectStoreLoggerHparams else Callback
    assert isinstance(construct_from_yaml(constructor, yaml_dict=yaml_dict), expected)


@pytest.mark.parametrize('cb_cls', get_cbs_and_marks(callbacks=True, loggers=True, profilers=True))
def test_callback_in_registry(cb_cls: Type[Callback]):
    # All callbacks, except for the ObjectStoreLogger and profiling callbacks, should appear in the registry
    # The ObjectStoreLogger has its own hparams class, and the profiling callbacks should not be instantiated
    # directly by the user
    if cb_cls is ObjectStoreLogger:
        item = ObjectStoreLoggerHparams
    else:
        item = cb_cls
    if cb_cls in [TorchProfiler, SystemProfiler, JSONTraceHandler, TraceHandler]:
        pytest.skip(
            f'Callback {cb_cls.__name__} does not have a registry entry as it should not be constructed directly')
    joint_registry = {**callback_registry, **logger_registry}
    assert_in_registry(item, registry=joint_registry)
