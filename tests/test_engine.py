# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Sequence
from unittest.mock import Mock

import pytest

from composer.algorithms import SelectiveBackprop
from composer.core import Event, engine
from composer.core.algorithm import Algorithm
from composer.core.state import State
from composer.loggers import Logger


@pytest.fixture
def always_match_algorithms():
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


def run_event(event: Event, state: State, logger: Logger):
    runner = engine.Engine(state, logger)
    return runner.run_event(event)


@pytest.mark.parametrize('event', list(Event))
class TestAlgorithms:

    def test_algorithms_always_called(self, event: Event, dummy_state: State, always_match_algorithms: List[Algorithm],
                                      dummy_logger: Logger):
        dummy_state.algorithms = always_match_algorithms
        _ = run_event(event, dummy_state, dummy_logger)
        for algo in always_match_algorithms:
            algo.apply.assert_called_once()
            algo.match.assert_called_once()

    def test_algorithms_never_called(self, event: Event, dummy_state: State, never_match_algorithms: List[Algorithm],
                                     dummy_logger: Logger):
        dummy_state.algorithms = never_match_algorithms
        _ = run_event(event, dummy_state, dummy_logger)
        for algo in never_match_algorithms:
            algo.apply.assert_not_called()
            algo.match.assert_called_once()

    def test_engine_trace_all(self, event: Event, dummy_state: State, always_match_algorithms: List[Algorithm],
                              dummy_logger: Logger):
        dummy_state.algorithms = always_match_algorithms
        trace = run_event(event, dummy_state, dummy_logger)

        assert all([tr.run for tr in trace.values()])

    def test_engine_trace_never(self, event: Event, dummy_state: State, never_match_algorithms: List[Algorithm],
                                dummy_logger: Logger):
        dummy_state.algorithms = never_match_algorithms
        trace = run_event(event, dummy_state, dummy_logger)

        assert all([tr.run is False for tr in trace.values()])


@pytest.mark.parametrize('event', [
    Event.EPOCH_START,
    Event.BEFORE_LOSS,
    Event.BEFORE_BACKWARD,
])
def test_engine_lifo_first_in(event: Event, dummy_state: State, dummy_logger: Logger,
                              always_match_algorithms: List[Algorithm]):
    dummy_state.algorithms = always_match_algorithms
    trace = run_event(event, dummy_state, dummy_logger)
    order = [tr.order for tr in trace.values()]
    expected_order = [tr.exit_code for tr in trace.values()]  # use exit_code to uniquely label algos

    assert order == expected_order


@pytest.mark.parametrize('event', [
    Event.AFTER_LOSS,
    Event.AFTER_BACKWARD,
    Event.BATCH_END,
])
def test_engine_lifo_last_out(event: Event, dummy_state: State, always_match_algorithms: List[Algorithm],
                              dummy_logger: Logger):
    dummy_state.algorithms = always_match_algorithms
    trace = run_event(event, dummy_state, dummy_logger)
    order = [tr.order for tr in trace.values()]
    expected_order = list(reversed([tr.exit_code for tr in trace.values()]))

    assert order == expected_order


def test_engine_with_selective_backprop(always_match_algorithms: Sequence[Algorithm], dummy_logger: Logger,
                                        dummy_state: State):
    sb = SelectiveBackprop(start=0.5, end=0.9, keep=0.5, scale_factor=0.5, interrupt=2)
    sb.apply = Mock(return_value='sb')
    sb.match = Mock(return_value=True)

    event = Event.INIT  # doesn't matter for this test

    algorithms = list(always_match_algorithms[0:2]) + [sb] + list(always_match_algorithms[2:])
    dummy_state.algorithms = algorithms

    trace = run_event(event, dummy_state, dummy_logger)

    expected = ['sb', 0, 1, 2, 3, 4]
    actual = [tr.exit_code for tr in trace.values()]

    assert actual == expected
