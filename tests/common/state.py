from composer.core import State
from tests.common.compare import deep_compare


def assert_state_equivalent(state1: State, state2: State):
    """Assert that ``state1`` is equivalent to ``state2``."""
    assert state1.serialized_attributes == state2.serialized_attributes
    assert state1.is_model_deepspeed == state2.is_model_deepspeed

    # Using a loose tolerance for GPU states as GPU determinism does not work properly
    is_gpu = next(state1.model.parameters()).device.type == "cuda"
    atol = 0.1 if is_gpu else 0.0
    rtol = 0.1 if is_gpu else 0.0

    # Compare the state dicts
    deep_compare(state1.state_dict(), state2.state_dict(), atol=atol, rtol=rtol)
