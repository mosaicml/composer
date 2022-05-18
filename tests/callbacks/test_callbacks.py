# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict

import pytest
from torch.utils.data import DataLoader

from composer.callbacks.callback_hparams import callback_registry
from composer.core import Event
from composer.core.callback import Callback
from composer.core.engine import Engine
from composer.core.state import State
from composer.loggers import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.logger_hparams import ObjectStoreLoggerHparams, logger_registry
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.profiler import JSONTraceHandler, SystemProfiler, TorchProfiler
from composer.profiler.profiler import Profiler
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.trace_handler import TraceHandler
from composer.trainer.trainer import Trainer
from tests.callbacks.callback_settings import get_callback_parametrization, get_callback_registry_parameterization
from tests.common import EventCounterCallback
from tests.common.datasets import RandomClassificationDataset
from tests.common.hparams import assert_is_constructable_from_yaml, assert_registry_contains_entry
from tests.common.models import SimpleModel


def test_callbacks_map_to_events():
    # callback methods must be 1:1 mapping with events
    # exception for private methods
    cb = Callback()
    excluded_methods = ["state_dict", "load_state_dict", "run_event", "close", "post_close"]
    methods = set(m for m in dir(cb) if (m not in excluded_methods and not m.startswith("_")))
    event_names = set(e.value for e in Event)
    assert methods == event_names


@pytest.mark.parametrize('event', list(Event))
def test_run_event_callbacks(event: Event, dummy_state: State):
    callback = EventCounterCallback()
    logger = Logger(dummy_state)
    dummy_state.callbacks = [callback]
    engine = Engine(state=dummy_state, logger=logger)

    engine.run_event(event)

    assert callback.event_to_num_calls[event] == 1


@pytest.mark.parametrize('cb_cls,cb_kwargs', get_callback_parametrization())
class TestCallbacks:

    @classmethod
    def setup_class(cls):
        pytest.importorskip("wandb", reason="WandB is optional.")

    def test_callback_is_constructable(self, cb_cls: Callable[..., Callback], cb_kwargs: Dict[str, Any]):
        cb = cb_cls(**cb_kwargs)
        assert isinstance(cb_cls, type)
        assert isinstance(cb, cb_cls)

    def test_callback_in_registry(self, cb_cls: Callable, cb_kwargs: Dict[str, Any]):
        # All callbacks, except for the ObjectStoreLogger and profiling callbacks, should appear in the registry
        # The ObjectStoreLogger has its own hparams class, and the profiling callbacks should not be instantiated
        # directly by the user
        if cb_cls is ObjectStoreLogger:
            cb_cls = ObjectStoreLoggerHparams
        if cb_cls in [TorchProfiler, SystemProfiler, JSONTraceHandler, TraceHandler]:
            pytest.skip(
                f"Callback {cb_cls.__name__} does not have a registry entry as it should not be constructed directly")
        joint_registry = {**callback_registry, **logger_registry}
        assert_registry_contains_entry(cb_cls, registry=joint_registry)

    def test_multiple_fit_start_and_end(self, cb_cls: Callable[..., Callback], cb_kwargs: Dict[str, Any],
                                        dummy_state: State):
        """Test that callbacks do not crash when Event.FIT_START and Event.FIT_END is called multiple times."""
        dummy_state.callbacks.append(cb_cls(**cb_kwargs))
        dummy_state.profiler = Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[])
        dummy_state.profiler.bind_to_state(dummy_state)

        logger = Logger(dummy_state)
        engine = Engine(state=dummy_state, logger=logger)

        engine.run_event(Event.INIT)  # always runs just once per engine

        engine.run_event(Event.FIT_START)
        engine.run_event(Event.FIT_END)

        engine.run_event(Event.FIT_START)
        engine.run_event(Event.FIT_END)

    def test_idempotent_close(self, cb_cls: Callable[..., Callback], cb_kwargs: Dict[str, Any], dummy_state: State):
        """Test that callbacks do not crash when .close() and .post_close() are called multiple times."""
        dummy_state.callbacks.append(cb_cls(**cb_kwargs))
        dummy_state.profiler = Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[])
        dummy_state.profiler.bind_to_state(dummy_state)

        logger = Logger(dummy_state)
        engine = Engine(state=dummy_state, logger=logger)

        engine.run_event(Event.INIT)  # always runs just once per engine
        engine.close()
        engine.close()

    def test_multiple_init_and_close(self, cb_cls: Callable[..., Callback], cb_kwargs: Dict[str, Any],
                                     dummy_state: State):
        """Test that callbacks do not crash when INIT/.close()/.post_close() are called multiple times in that order."""
        dummy_state.callbacks.append(cb_cls(**cb_kwargs))
        dummy_state.profiler = Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[])
        dummy_state.profiler.bind_to_state(dummy_state)

        logger = Logger(dummy_state)
        engine = Engine(state=dummy_state, logger=logger)

        engine.run_event(Event.INIT)
        engine.close()
        # For good measure, also test idempotent close, in case if there are edge cases with a second call to INIT
        engine.close()

        # Create a new engine, since the engine does allow events to run after it has been closed
        engine = Engine(state=dummy_state, logger=logger)
        engine.close()
        # For good measure, also test idempotent close, in case if there are edge cases with a second call to INIT
        engine.close()


@pytest.mark.parametrize('cb_cls,cb_kwargs', get_callback_parametrization())
@pytest.mark.parametrize('grad_accum', [1, 2])
@pytest.mark.filterwarnings(r'ignore:The profiler is enabled:UserWarning')
class TestCallbackTrains:

    def _get_trainer(self, cb: Callback, grad_accum: int):
        loggers = cb if isinstance(cb, LoggerDestination) else None
        callbacks = cb if not isinstance(cb, LoggerDestination) else None

        return Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset(), batch_size=2),
            eval_dataloader=DataLoader(RandomClassificationDataset(), batch_size=2),
            compute_training_metrics=True,
            max_duration=2,
            grad_accum=grad_accum,
            callbacks=callbacks,
            loggers=loggers,
            profiler=Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[]),
        )

    def test_trains(self, cb_cls: Callable[..., Callback], cb_kwargs: Dict[str, Any], grad_accum: int):
        cb = cb_cls(**cb_kwargs)
        trainer = self._get_trainer(cb, grad_accum)
        trainer.fit()

    def test_trains_multiple_calls(self, cb_cls: Callable[..., Callback], cb_kwargs: Dict[str, Any], grad_accum: int):
        """
        Tests that training with multiple fits complete.
        Note: future functional tests should test for
        idempotency (e.g functionally)
        """
        cb = cb_cls(**cb_kwargs)
        trainer = self._get_trainer(cb, grad_accum)
        trainer.fit()

        assert trainer.state.max_duration is not None
        trainer.state.max_duration *= 2

        trainer.fit()


@pytest.mark.parametrize('constructor,yaml_dict', get_callback_registry_parameterization())
def test_callback_hparams_is_constructable(
    constructor: Callable,
    yaml_dict: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    # The ObjectStoreLogger needs the KEY_ENVIRON set
    if constructor is ObjectStoreLoggerHparams:
        monkeypatch.setenv('KEY_ENVIRON', '.')
    assert_is_constructable_from_yaml(constructor, yaml_dict=yaml_dict, expected=Callback)
