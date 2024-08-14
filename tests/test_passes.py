# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence
from unittest.mock import Mock

import pytest

from composer import Algorithm, Engine, Event, Logger, State, Trainer
from composer.algorithms import LowPrecisionLayerNorm, SelectiveBackprop
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import LowPrecisionLayerNorm
from composer.core.passes import sort_to_back, sort_to_front
from composer.devices import DeviceCPU
from tests.common import SimpleModel

from .test_engine import run_event


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


def test_register_pass(dummy_state, dummy_logger):

    dummy_algorithm = Mock()
    dummy_algorithm.match.return_value = True
    dummy_algorithm.apply.return_value = 'dummy'

    def insert_dummy_algorithm(algorithms, event):
        algorithms.append(dummy_algorithm)
        return algorithms

    engine = Engine(dummy_state, dummy_logger)
    engine.register_pass(insert_dummy_algorithm)

    trace = engine.run_event(Event.INIT)

    assert 'dummy' in [tr.exit_code for tr in trace.values()]


class TestLIFOPass:

    @pytest.mark.parametrize(
        'event',
        [
            Event.BEFORE_LOSS,
            Event.BEFORE_BACKWARD,
        ],
    )
    def test_lifo_first_in(
        self,
        event: Event,
        dummy_state: State,
        dummy_logger: Logger,
        always_match_algorithms: list[Algorithm],
    ):
        dummy_state.algorithms = always_match_algorithms
        trace = run_event(event, dummy_state, dummy_logger)
        order = [tr.order for tr in trace.values()]
        expected_order = [tr.exit_code for tr in trace.values()]  # use exit_code to uniquely label algos

        assert order == expected_order

    @pytest.mark.parametrize(
        'event',
        [
            Event.AFTER_LOSS,
            Event.AFTER_BACKWARD,
        ],
    )
    def test_lifo_last_out(
        self,
        event: Event,
        dummy_state: State,
        always_match_algorithms: list[Algorithm],
        dummy_logger: Logger,
    ):
        dummy_state.algorithms = always_match_algorithms
        trace = run_event(event, dummy_state, dummy_logger)
        order = [tr.order for tr in trace.values()]
        expected_order = list(reversed([tr.exit_code for tr in trace.values()]))

        assert order == expected_order


class TestAlgorithmOrderingPasses:

    @pytest.mark.parametrize('algorithm_cls', [LowPrecisionLayerNorm])
    def test_algorithm_last(
        self,
        algorithm_cls: type[Algorithm],
        always_match_algorithms: list[Algorithm],
        dummy_logger: Logger,
        dummy_state: State,
    ):
        algorithm = algorithm_cls()
        algorithm.apply = Mock(return_value='algo')
        algorithm.match = Mock(return_value=True)

        algortihms = always_match_algorithms[0:2] + [algorithm] + always_match_algorithms[2:]
        dummy_state._algorithms = algortihms

        trace = run_event(Event.INIT, dummy_state, dummy_logger)

        expected = [0, 1, 2, 3, 4, 'algo']
        actual = [tr.exit_code for tr in trace.values()]

        assert actual == expected

    @pytest.mark.parametrize('algorithm_cls', [SelectiveBackprop])
    def test_algorithm_first(
        self,
        algorithm_cls: type[Algorithm],
        always_match_algorithms: list[Algorithm],
        dummy_logger: Logger,
        dummy_state: State,
    ):

        algorithm = algorithm_cls()
        algorithm.apply = Mock(return_value='algo')
        algorithm.match = Mock(return_value=True)

        algorithms = always_match_algorithms[0:2] + [algorithm] + always_match_algorithms[2:]
        dummy_state._algorithms = algorithms

        trace = run_event(Event.INIT, dummy_state, dummy_logger)

        expected = ['algo', 0, 1, 2, 3, 4]
        actual = [tr.exit_code for tr in trace.values()]

        assert actual == expected


class TestSortHelpers:

    def test_sort_to_back(self):
        lst = [1, 'a', 'c', 2, 3.0]
        assert sort_to_back(lst, int) == ['a', 'c', 3.0, 1, 2]

    def test_sort_to_front(self):
        lst = [1, 'a', 'c', 2, 3.0]
        assert sort_to_front(lst, int) == [1, 2, 'a', 'c', 3.0]


def get_default_passes():
    state = State(model=SimpleModel(), device=DeviceCPU(), rank_zero_seed=42, run_name='test_chungoose')
    engine = Engine(state, Logger(state))
    return engine.algorithm_passes


def get_custom_pass():

    def sort_by_name(algorithms: Sequence[Algorithm], event: Event) -> Sequence[Algorithm]:
        return sorted(algorithms, key=lambda x: type(x).__name__)

    return sort_by_name


sort_by_name = get_custom_pass()  # Generate pass object so we can use same ref in tests


class TestTrainerArg:

    @pytest.mark.parametrize(
        'algorithm_passes,expected_passes',
        [
            [None, get_default_passes()],
            [sort_by_name, get_default_passes() + [sort_by_name]],
            [[sort_by_name], get_default_passes() + [sort_by_name]],
            [[sort_by_name, 0], [sort_by_name] + get_default_passes()],  # type: ignore
            [(sort_by_name, 0), [sort_by_name] + get_default_passes()],  # type: ignore
            [[(sort_by_name, 0)], [sort_by_name] + get_default_passes()],  # type: ignore
        ],
    )
    def test_add_pass(self, algorithm_passes, expected_passes):
        trainer = Trainer(model=SimpleModel(), algorithm_passes=algorithm_passes)
        assert trainer.engine.algorithm_passes == expected_passes
