# Contributing algorithm tests

Algorithm-based tests should follow a similar template, as shown below:

```
import pytest


@pytest.fixture
def state(minimal_state):
    # make any test-specific needed modifications
    # e.g. adding a conv model, or changing the dataloader
    return minimal_state


"""Every algorithm test should have the functional and the algorithm
usage demonstrated below in the tests.
"""


def test_myalgo_functional():
    ...


def test_myalgo_algorithm(state, empty_logger):
    ...


""" Results from logging and hparams initialization should also be tested.
"""


def test_myalgo_logging(state):
    """Test that the logging is as expected.

    Example:

        logger_mock = Mock()
        algorithm = AlgorithmThatLogsSomething()
        algorithm.apply(Event.INIT, state, logger=logger_mock)

        logger_mock.log_hyperparameters.assert_called_one_with({
            'some_key': some_value
        })
    """
```

The above is a minimal set, the rest of the tests will vary depending on the exact algorithm under test.
