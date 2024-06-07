# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

import composer
from composer.core import Engine, Event
from composer.core.algorithm import Algorithm
from composer.core.callback import Callback
from composer.core.state import State
from composer.loggers import Logger, LoggerDestination
from tests.common.events import EventCounterCallback


@pytest.fixture
def always_match_algorithms():
    return [
        Mock(
            **{
                'match.return.value': True,
                'apply.return_value': n,  # return encodes order
                'interpolate_loss': False,
            },
        ) for n in range(5)
    ]


@pytest.fixture()
def dummy_logger(dummy_state: State):
    return Logger(dummy_state)


@pytest.fixture
def never_match_algorithms():
    attrs = {'match.return_value': False}
    return [Mock(**attrs) for _ in range(5)]


def run_event(event: Event, state: State, logger: Logger):
    runner = Engine(state, logger)
    return runner.run_event(event)


class DummyCallback(Callback):

    def __init__(self, file_path):
        self.file_path = file_path

    def init(self, state: State, logger: Logger):
        with open(self.file_path, 'a') as f:
            f.write('init callback, ')

    def batch_end(self, state: State, logger: Logger):
        with open(self.file_path, 'a') as f:
            f.write('on_batch_end callback, ')


class DummyLoggerDestination(LoggerDestination):

    def __init__(self, file_path):
        self.file_path = file_path

    def init(self, state: State, logger: Logger):
        with open(self.file_path, 'a') as f:
            f.write('init logger, ')

    def batch_end(self, state: State, logger: Logger):
        with open(self.file_path, 'a') as f:
            f.write('on_batch_end logger, ')


def test_engine_runs_callbacks_in_correct_order(dummy_state, tmp_path):
    file_path = tmp_path / Path('event_check.txt')
    dummy_state.callbacks = [DummyCallback(file_path), DummyLoggerDestination(file_path)]
    logger = Logger(dummy_state)
    engine = Engine(dummy_state, logger)
    engine.run_event(Event.INIT)
    engine.run_event(Event.BATCH_END)
    engine.run_event(Event.EPOCH_END)
    engine.close()
    expected_lines = ['init logger, init callback, on_batch_end callback, on_batch_end logger, ']
    with open(file_path, 'r') as f:
        actual_lines = f.readlines()
    assert expected_lines == actual_lines


@pytest.mark.parametrize('event', list(Event))
class TestAlgorithms:

    def test_algorithms_always_called(
        self,
        event: Event,
        dummy_state: State,
        always_match_algorithms: list[Algorithm],
        dummy_logger: Logger,
    ):
        dummy_state.algorithms = always_match_algorithms
        _ = run_event(event, dummy_state, dummy_logger)
        for algo in always_match_algorithms:
            algo.apply.assert_called_once()
            algo.match.assert_called_once()

    def test_algorithms_never_called(
        self,
        event: Event,
        dummy_state: State,
        never_match_algorithms: list[Algorithm],
        dummy_logger: Logger,
    ):
        dummy_state.algorithms = never_match_algorithms
        _ = run_event(event, dummy_state, dummy_logger)
        for algo in never_match_algorithms:
            algo.apply.assert_not_called()
            algo.match.assert_called_once()

    def test_engine_trace_all(
        self,
        event: Event,
        dummy_state: State,
        always_match_algorithms: list[Algorithm],
        dummy_logger: Logger,
    ):
        dummy_state.algorithms = always_match_algorithms
        trace = run_event(event, dummy_state, dummy_logger)

        assert all(tr.run for tr in trace.values())

    def test_engine_trace_never(
        self,
        event: Event,
        dummy_state: State,
        never_match_algorithms: list[Algorithm],
        dummy_logger: Logger,
    ):
        dummy_state.algorithms = never_match_algorithms
        trace = run_event(event, dummy_state, dummy_logger)

        assert all(tr.run is False for tr in trace.values())


def test_engine_is_dead_after_close(dummy_state: State, dummy_logger: Logger):
    # Create the trainer and run an event
    engine = Engine(dummy_state, dummy_logger)
    engine.run_event(Event.INIT)

    # Close it
    engine.close()

    # Assert it complains if you try to run another event
    with pytest.raises(RuntimeError):
        engine.run_event(Event.FIT_START)


class IsClosedCallback(Callback):

    def __init__(self) -> None:
        self.is_closed = True

    def init(self, state: State, logger: Logger) -> None:
        assert self.is_closed
        self.is_closed = False

    def close(self, state: State, logger: Logger) -> None:
        self.is_closed = True


def test_engine_closes_on_del(dummy_state: State, dummy_logger: Logger):
    # Create the trainer and run an event
    is_closed_callback = IsClosedCallback()
    dummy_state.callbacks.append(is_closed_callback)
    engine = Engine(dummy_state, dummy_logger)
    engine.run_event(Event.INIT)

    # Assert that there is just 2 -- once above, and once as the arg temp reference
    assert sys.getrefcount(engine) == 2

    # Implicitly close the engine
    del engine

    # Assert it is closed
    assert is_closed_callback.is_closed


class DummyTrainer:
    """Helper to simulate what the trainer does w.r.t. events"""

    def __init__(self, state: State, logger: Logger) -> None:
        self.engine = Engine(state, logger)
        self.engine.run_event(Event.INIT)

    def close(self):
        self.engine.close()


def test_engine_triggers_close_only_once(dummy_state: State, dummy_logger: Logger):
    # Create the trainer and run an event
    is_closed_callback = IsClosedCallback()
    dummy_state.callbacks.append(is_closed_callback)

    # Create the trainer
    trainer = DummyTrainer(dummy_state, dummy_logger)

    # Close the trainer
    trainer.close()

    # Assert it is closed
    assert is_closed_callback.is_closed

    # Create a new trainer with the same callback. Should implicitly trigger __del__ AFTER
    # AFTER DummyTrainer was constructed
    trainer = DummyTrainer(dummy_state, dummy_logger)

    # Assert it is open
    assert not is_closed_callback.is_closed


def test_engine_errors_if_previous_trainer_was_not_closed(dummy_state: State, dummy_logger: Logger):
    # Create the trainer and run an event
    is_closed_callback = IsClosedCallback()
    dummy_state.callbacks.append(is_closed_callback)

    # Create the trainer
    _ = DummyTrainer(dummy_state, dummy_logger)

    # Assert the callback is open
    assert not is_closed_callback.is_closed

    # Create a new trainer with the same callback. Should raise an exception
    # because trainer.close() was not called before
    with pytest.raises(
        RuntimeError,
        match=r'Cannot create a new trainer with an open callback or logger from a previous trainer',
    ):
        DummyTrainer(dummy_state, dummy_logger)


def check_output(proc: subprocess.CompletedProcess):
    # Check the subprocess output, and raise an exception with the stdout/stderr dump if there was a non-zero exit
    # The `check=True` flag available in `subprocess.run` does not print stdout/stderr
    if proc.returncode == 0:
        return
    error_msg = textwrap.dedent(
        f"""\
        Command {proc.args} failed with exit code {proc.returncode}.
        ----Begin stdout----
        {proc.stdout}
        ----End stdout------
        ----Begin stderr----
        {proc.stderr}
        ----End stderr------""",
    )

    raise RuntimeError(error_msg)


@pytest.mark.parametrize('exception', [True, False])
def test_engine_closes_on_atexit(exception: bool):
    # Running this test via a subprocess, as atexit() must trigger

    code = textwrap.dedent(
        """\
    from composer import Trainer, Callback
    from tests.common import SimpleModel

    class CallbackWithConditionalCloseImport(Callback):
        def post_close(self):
            import requests

    model = SimpleModel(3, 10)
    cb = CallbackWithConditionalCloseImport()
    trainer = Trainer(
        model=model,
        callbacks=[cb],
        max_duration="1ep",
        train_dataloader=None,
    )
    """,
    )
    if exception:
        # Should raise an exception, since no dataloader was provided
        code += 'trainer.fit()'

    git_root_dir = os.path.join(os.path.dirname(composer.__file__), '..')
    proc = subprocess.run(['python', '-c', code], cwd=git_root_dir, text=True, capture_output=True)
    if exception:
        # manually validate that there was no a conditional import exception
        assert 'ImportError: sys.meta_path is None, Python is likely shutting down' not in proc.stderr
    else:
        check_output(proc)


def test_logging(
    caplog: pytest.LogCaptureFixture,
    dummy_state: State,
    dummy_logger: Logger,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that engine logs statements as expected"""
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=Engine.__module__):
        # Include a callback, since most logging happens around callback events
        dummy_state.callbacks = [EventCounterCallback()]

        monkeypatch.setenv('ENGINE_DEBUG', '1')
        engine = Engine(dummy_state, dummy_logger)
        engine.run_event('INIT')
        engine.close()

        # Validate that we have the expected log entries
        assert caplog.record_tuples == [
            ('composer.core.engine', 10, '[ep=0][ba=0][event=INIT]: Running event'),
            ('composer.core.engine', 10, '[ep=0][ba=0][event=INIT]: Running callback EventCounterCallback'),
            ('composer.core.engine', 10, 'Closing the engine.'),
            ('composer.core.engine', 10, 'Closing callback EventCounterCallback'),
            ('composer.core.engine', 10, 'Post-closing callback EventCounterCallback'),
            ('composer.core.engine', 10, 'Engine closed.'),
        ]


def _worker():
    import composer.core.engine
    importlib.reload(composer.core.engine)


def test_graceful_fallback_when_signal_handler_cannot_be_set():
    # https://github.com/mosaicml/composer/issues/3151#issue-2205981731
    t = threading.Thread(target=_worker)
    t.start()
    t.join()
