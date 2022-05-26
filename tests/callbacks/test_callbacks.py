# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type, Union

import pytest
from torch.utils.data import DataLoader

from composer.callbacks.callback_hparams import CallbackHparams, callback_registry
from composer.core import Callback, Engine, Event, State
from composer.loggers import Logger, LoggerDestination, ObjectStoreLogger
from composer.loggers.logger_hparams import LoggerDestinationHparams, ObjectStoreLoggerHparams, logger_registry
from composer.profiler import JSONTraceHandler, Profiler, ProfilerAction, SystemProfiler, TorchProfiler, TraceHandler
from composer.trainer import Trainer
from tests.callbacks.callback_settings import get_cb_hparams_and_marks, get_cb_kwargs, get_cbs_and_marks
from tests.common import EventCounterCallback, RandomClassificationDataset, SimpleModel
from tests.common.hparams import assert_in_registry, assert_yaml_loads


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


@pytest.mark.parametrize('cb_cls', get_cbs_and_marks())
class TestCallbacks:

    @classmethod
    def setup_class(cls):
        pytest.importorskip("wandb", reason="WandB is optional.")

    def test_callback_is_constructable(self, cb_cls: Type[Callback]):
        cb_kwargs = get_cb_kwargs(cb_cls)
        cb = cb_cls(**cb_kwargs)
        assert isinstance(cb_cls, type)
        assert isinstance(cb, cb_cls)

    @pytest.mark.xfail(reason="This test requires AutoYAHP, which will put the callbacks directly into the registry")
    def test_callback_in_registry(self, cb_cls: Type[Callback]):
        # All callbacks, except for the ObjectStoreLogger and profiling callbacks, should appear in the registry
        # The ObjectStoreLogger has its own hparams class, and the profiling callbacks should not be instantiated
        # directly by the user
        if cb_cls is ObjectStoreLogger:
            item = ObjectStoreLoggerHparams
        else:
            item = cb_cls
        if cb_cls in [TorchProfiler, SystemProfiler, JSONTraceHandler, TraceHandler]:
            pytest.skip(
                f"Callback {cb_cls.__name__} does not have a registry entry as it should not be constructed directly")
        joint_registry = {**callback_registry, **logger_registry}
        assert_in_registry(item, registry=joint_registry)

    def test_multiple_fit_start_and_end(self, cb_cls: Type[Callback], dummy_state: State):
        """Test that callbacks do not crash when Event.FIT_START and Event.FIT_END is called multiple times."""
        cb_kwargs = get_cb_kwargs(cb_cls)
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

    def test_idempotent_close(self, cb_cls: Type[Callback], dummy_state: State):
        """Test that callbacks do not crash when .close() and .post_close() are called multiple times."""
        cb_kwargs = get_cb_kwargs(cb_cls)
        dummy_state.callbacks.append(cb_cls(**cb_kwargs))
        dummy_state.profiler = Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[])
        dummy_state.profiler.bind_to_state(dummy_state)

        logger = Logger(dummy_state)
        engine = Engine(state=dummy_state, logger=logger)

        engine.run_event(Event.INIT)  # always runs just once per engine
        engine.close()
        engine.close()

    def test_multiple_init_and_close(self, cb_cls: Type[Callback], dummy_state: State):
        """Test that callbacks do not crash when INIT/.close()/.post_close() are called multiple times in that order."""
        cb_kwargs = get_cb_kwargs(cb_cls)
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


@pytest.mark.parametrize('cb_cls', get_cbs_and_marks())
@pytest.mark.parametrize('grad_accum', [1, 2])
@pytest.mark.filterwarnings(r'ignore:The profiler is enabled:UserWarning')
class TestCallbackTrains:

    def _get_trainer(self, cb: Callback, grad_accum: int):
        loggers = cb if isinstance(cb, LoggerDestination) else None
        callbacks = cb if not isinstance(cb, LoggerDestination) else None

        return Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(RandomClassificationDataset(size=4), batch_size=2),
            eval_dataloader=DataLoader(RandomClassificationDataset(size=4), batch_size=2),
            compute_training_metrics=True,
            max_duration=2,
            grad_accum=grad_accum,
            callbacks=callbacks,
            loggers=loggers,
            profiler=Profiler(schedule=lambda _: ProfilerAction.SKIP, trace_handlers=[]),
        )

    @pytest.mark.timeout(15)
    def test_trains(self, cb_cls: Type[Callback], grad_accum: int):
        cb_kwargs = get_cb_kwargs(cb_cls)
        cb = cb_cls(**cb_kwargs)
        trainer = self._get_trainer(cb, grad_accum)
        trainer.fit()

    @pytest.mark.timeout(15)
    def test_trains_multiple_calls(self, cb_cls: Type[Callback], grad_accum: int):
        """
        Tests that training with multiple fits complete.
        Note: future functional tests should test for
        idempotency (e.g functionally)
        """
        cb_kwargs = get_cb_kwargs(cb_cls)
        cb = cb_cls(**cb_kwargs)
        trainer = self._get_trainer(cb, grad_accum)
        trainer.fit()

        assert trainer.state.max_duration is not None
        trainer.state.max_duration *= 2

        trainer.fit()


@pytest.mark.parametrize('constructor', get_cb_hparams_and_marks())
def test_callback_hparams_is_constructable(
    constructor: Union[Type[Callback], Type[CallbackHparams], Type[LoggerDestinationHparams]],
    monkeypatch: pytest.MonkeyPatch,
):
    # The ObjectStoreLogger needs the KEY_ENVIRON set
    yaml_dict = get_cb_kwargs(constructor)
    if constructor is ObjectStoreLoggerHparams:
        monkeypatch.setenv('KEY_ENVIRON', '.')
    assert_yaml_loads(
        constructor,
        yaml_dict=yaml_dict,
        expected=(CallbackHparams, LoggerDestinationHparams),
    )
