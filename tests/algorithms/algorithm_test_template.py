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
    """ Test that the logging is as expected.

    Example:

        logger_mock = Mock()
        algorithm = AlgorithmThatLogsSomething()
        algorithm.apply(Event.INIT, state, logger=logger_mock)

        logger_mock.metric_fit.assert_called_one_with({
            'some_key': some_value
        })


    """


def test_myalgo_hparams():
    """ Test that the algorithm hparams can instantiate the class.

    Example:

        hparams = BlurPoolHparams(
            replace_convs=True,
            replace_maxpools=True
            blur_first=True
        )
        algorithm = hparams.initialize_object()
        assert isinstance(algorithm, BlurPool)

    """
    ...


# The above is the minimal set, the
# rest of the test suite will varying depending
# on the exact algorithm under test.
