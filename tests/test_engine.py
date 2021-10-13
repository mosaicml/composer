# Copyright 2021 MosaicML. All Rights Reserved.

from unittest.mock import Mock

import pytest

from composer.core import Event, engine


@pytest.fixture
def always_match_algorithms():
    attrs = {'match.return_value': True}
    return [
        Mock(**{
            'match.return.value': True,
            'apply.return_value': n,  # return encodes order
        }) for n in range(5)
    ]


@pytest.fixture
def never_match_algorithms():
    attrs = {'match.return_value': False}
    return [Mock(**attrs) for _ in range(5)]


def run_event(event, state, algorithms, logger):
    runner = engine.Engine(state, algorithms, logger)
    return runner.run_event(event)


@pytest.mark.parametrize('event', list(Event))
class TestAlgorithms:

    def test_algorithms_always_called(self, event, dummy_state, always_match_algorithms, dummy_logger):
        _ = run_event(event, dummy_state, always_match_algorithms, dummy_logger)
        for algo in always_match_algorithms:
            algo.apply.assert_called_once()
            algo.match.assert_called_once()

    def test_algorithms_never_called(self, event, dummy_state, never_match_algorithms, dummy_logger):
        _ = run_event(event, dummy_state, never_match_algorithms, dummy_logger)
        for algo in never_match_algorithms:
            algo.apply.assert_not_called()
            algo.match.assert_called_once()

    def test_engine_trace_all(self, event, dummy_state, always_match_algorithms, dummy_logger):
        trace = run_event(event, dummy_state, always_match_algorithms, dummy_logger)

        assert all([tr.run for tr in trace.values()])

    def test_engine_trace_never(self, event, dummy_state, never_match_algorithms, dummy_logger):
        trace = run_event(event, dummy_state, never_match_algorithms, dummy_logger)

        assert all([tr.run is False for tr in trace.values()])


@pytest.mark.parametrize(
    'event',
    [
        Event.TRAINING_START,
        Event.BEFORE_LOSS,
        Event.BEFORE_BACKWARD,
        Event.TRAINING_END,  # no before prefix, so run in normal order
    ])
def test_engine_lifo_first_in(event, dummy_state, always_match_algorithms, dummy_logger):
    trace = run_event(event, dummy_state, always_match_algorithms, dummy_logger)
    order = [tr.order for tr in trace.values()]
    expected_order = [tr.exit_code for tr in trace.values()]  # use exit_code to uniquely label algos

    assert order == expected_order


@pytest.mark.parametrize('event', [
    Event.AFTER_LOSS,
    Event.AFTER_BACKWARD,
])
def test_engine_lifo_last_out(event, dummy_state, always_match_algorithms, dummy_logger):
    trace = run_event(event, dummy_state, always_match_algorithms, dummy_logger)
    order = [tr.order for tr in trace.values()]
    expected_order = list(reversed([tr.exit_code for tr in trace.values()]))

    assert order == expected_order
