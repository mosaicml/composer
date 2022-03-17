
import pytest
from composer.datasets.utils import _default_split_batch

batches = ["dummy_tensor_batch", "dummy_tuple_batch", "dummy_dict_batch", "dummy_maskrcnn_batch"]


@pytest.mark.usefixtures(", ".join(batches))
@pytest.mark.parametrize("batch", batches)
def test_default_split_without_error(batch, request):
    batch = request.getfixturevalue(batch)
    _default_split_batch(batch, num_microbatches=3)


@pytest.mark.usefixtures(", ".join(batches))
@pytest.mark.parametrize("batch", batches)
def test_default_split_num_microbatches(batch, request):
    batch = request.getfixturevalue(batch)
    microbatch = _default_split_batch(batch, num_microbatches=3)
    assert len(microbatch) == 3


