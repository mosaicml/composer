from typing import Mapping

import pytest

from composer.core.data_spec import _default_split_batch, _split_list
from tests.common import dummy_batches, dummy_maskrcnn_batch, dummy_tuple_batch, dummy_tuple_batch_long


@pytest.mark.parametrize("batch", dummy_batches(12))
def test_split_without_error(batch):
    _default_split_batch(batch, num_microbatches=3)


@pytest.mark.parametrize("batch", [dummy_tuple_batch(12)])
def test_split_tuple(batch):
    microbatches = _default_split_batch(batch, num_microbatches=3)
    # should be 3 microbatches of size 4 tensors pairs
    # should split into [(x, y), (x, y), (x, y)]
    assert len(microbatches[0]) == 2


@pytest.mark.parametrize("batch", [dummy_tuple_batch_long(12)])
def test_split_tuple_long(batch):
    microbatches = _default_split_batch(batch, num_microbatches=3)
    assert len(microbatches[0]) == 4


@pytest.mark.parametrize("batch", [dummy_maskrcnn_batch(12)])
def test_split_maskrcnn(batch):
    microbatches = _split_list(batch, num_microbatches=3)
    assert len(microbatches) == 3


@pytest.mark.parametrize("batch", dummy_batches(12))
def test_num_micro_batches(batch):
    microbatch = _default_split_batch(batch, num_microbatches=3)
    assert len(microbatch) == 3


@pytest.mark.parametrize("batch", dummy_batches(5))
def test_odd_batch_sizes(batch):
    microbatch = _default_split_batch(batch, num_microbatches=3)
    # should split into [len(2), len(2), len(2)]
    last_microbatch = microbatch[-1]
    assert len(microbatch) == 3
    if isinstance(last_microbatch, Mapping):
        assert len(last_microbatch['image']) == 1
        assert len(last_microbatch['target']) == 1
    if isinstance(last_microbatch, tuple):
        assert len(last_microbatch[0]) == 1
    if isinstance(last_microbatch, list):
        assert len(last_microbatch) == 1


@pytest.mark.parametrize("batch", dummy_batches(1))
def test_batch_size_less_than_num_microbatches(batch):
    with pytest.raises(ValueError):
        _default_split_batch(batch, num_microbatches=3)
