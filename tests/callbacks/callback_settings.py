from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence

import pytest

import composer.callbacks
import composer.loggers
import composer.profiler
from composer import Callback
from composer.callbacks import EarlyStopper, MemoryMonitor, ThresholdStopper
from composer.callbacks.callback_hparams import (EarlyStopperHparams, MLPerfCallbackHparams, ThresholdStopperHparams,
                                                 callback_registry)
from composer.callbacks.mlperf import MLPerfCallback
from composer.loggers import ObjectStoreLogger, WandBLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.logger_hparams import ObjectStoreLoggerHparams, logger_registry
from composer.loggers.progress_bar_logger import ProgressBarLogger
from tests.common import get_all_subclasses_in_module

try:
    import wandb
    _WANDB_INSTALLED = True
    del wandb  # unused
except ImportError:
    _WANDB_INSTALLED = False

try:
    import mlperf_logging
    _MLPERF_INSTALLED = True
    del mlperf_logging
except ImportError:
    _MLPERF_INSTALLED = False


class _CallbackConfig(NamedTuple):
    kwargs: Optional[Dict[str, Any]] = None
    marks: Sequence[pytest.MarkDecorator] = ()


_default_settings: Dict[Callable, _CallbackConfig] = {
    ObjectStoreLogger:
        _CallbackConfig(
            kwargs={
                'use_procs': False,
                'num_concurrent_uploads': 1,
                'provider': 'local',
                'container': '.',
                'provider_kwargs': {
                    'key': '.',
                },
            },
            marks=[
                pytest.mark.filterwarnings(
                    # post_close might not be called if being used outside of the trainer
                    r'ignore:Implicitly cleaning up:ResourceWarning')
            ],
        ),
    MemoryMonitor:
        _CallbackConfig(
            kwargs={},
            marks=[pytest.mark.gpu],
        ),
    ThresholdStopper:
        _CallbackConfig(kwargs={
            'monitor': 'Accuracy',
            'dataloader_label': 'train',
            'threshold': 0.99,
        },),
    EarlyStopper:
        _CallbackConfig(kwargs={
            'monitor': 'Accuracy',
            'dataloader_label': 'train',
        },),
    MLPerfCallback:
        _CallbackConfig(
            kwargs={
                'root_folder': '.',
                'index': 0,
            },
            marks=[pytest.mark.skipif(not _MLPERF_INSTALLED, reason="MLPerf is optional")],
        ),
    WandBLogger:
        _CallbackConfig(
            kwargs={},
            marks=[
                pytest.mark.filterwarnings(r'ignore:unclosed file:ResourceWarning'),
                pytest.mark.skipif(not _WANDB_INSTALLED, reason="Wandb is optional"),
            ],
        ),
    ProgressBarLogger:
        _CallbackConfig(marks=[
            pytest.mark.filterwarnings(
                r"ignore:Specifying the ProgressBarLogger via `loggers` is deprecated:DeprecationWarning")
        ],),
    ObjectStoreLoggerHparams:
        _CallbackConfig(
            kwargs={
                'object_store_hparams': {
                    'provider': 'local',
                    'container': '.',
                    'key_environ': 'KEY_ENVIRON',
                },
                'use_procs': False,
                'num_concurrent_uploads': 1,
            },
            marks=[
                pytest.mark.filterwarnings(
                    # post_close might not be called if being used outside of the trainer
                    r'ignore:Implicitly cleaning up:ResourceWarning')
            ],
        ),
}

# Manually inject some of the hparam classes into the settings.
# TODO(ravi): Remove this with autoyahp
_default_settings[ThresholdStopperHparams] = _default_settings[ThresholdStopper]
_default_settings[EarlyStopperHparams] = _default_settings[EarlyStopper]
_default_settings[MLPerfCallbackHparams] = _default_settings[MLPerfCallback]


def _to_pytest_param(impl: Callable):
    if impl not in _default_settings:
        return pytest.param(impl, {})
    else:
        kwargs = _default_settings[impl].kwargs
        if kwargs is None:
            kwargs = {}
        marks = _default_settings[impl].marks
        return pytest.param(impl, kwargs, marks=marks)


def get_callback_parametrization():
    """Returns a list of :class:`pytest.mark.param` objects for all :class:`.Callback`.
    The callbacks are correctly annotated with ``skipif`` marks for optional dependencies
    and ``filterwarning`` marks for any warnings that might be emitted and are safe to ignore

    This function is meant to be used like this::

        import pytest
        from tests.callbacks.callback_settings import get_callback_parametrization

        @pytest.mark.parametrize("cb_cls,cb_kwargs",get_callback_parametrization())
        def test_something(cb_cls: Callable[..., Callback], cb_kwargs: Dict[str, Any]):
            cb = cb_cls(**cb_kwargs)
            assert isinstance(cb, Callback)
    """
    implementations = [
        *get_all_subclasses_in_module(composer.callbacks, Callback),
        *get_all_subclasses_in_module(composer.loggers, LoggerDestination),
        *get_all_subclasses_in_module(composer.profiler, Callback),
    ]
    ans = [_to_pytest_param(impl) for impl in implementations]
    return ans


def get_callback_registry_parameterization():
    """Returns a list of :class:`pytest.mark.param` objects for all ``callback_registry``
    and ``logger_registry``entries.

    The callbacks are correctly annotated with ``skipif`` marks for optional dependencies
    and ``filterwarning`` marks for any warnings that might be emitted and are safe to ignore

    This function is meant to be used like this::

        import pytest
        from tests.common import assert_is_constructable_from_yaml
        from tests.callbacks.callback_settings import get_callback_registry_parameterization

        @pytest.mark.parametrize("constructor,yaml_dict",get_callback_registry_parameterization())
        def test_something(constructor: Callable, yaml_dict: Dict[str, Any]):
            assert_is_constructable_from_yaml(constructor, yaml_dict=yaml_dict)
    """
    implementations = [
        *callback_registry.values(),
        *logger_registry.values(),
    ]
    ans = [_to_pytest_param(impl) for impl in implementations]
    return ans
