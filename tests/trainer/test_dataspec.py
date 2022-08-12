# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from composer.core.data_spec import DataSpec
from tests.common import RandomClassificationDataset

N = 128


class TestDefaultGetNumSamples:

    @pytest.fixture
    def dataspec(self):
        dataloader = DataLoader(RandomClassificationDataset())
        return DataSpec(dataloader=dataloader)

    # yapf: disable
    @pytest.mark.parametrize('batch', [
        {'a': torch.rand(N, 8), 'b': torch.rand(N, 64)},  # dict
        [{'a': torch.rand(N, 8)}, {'c': torch.rand(N, 64)}],  # list of dict
        (torch.rand(N, 8), torch.rand(N, 64)),  # tuple
        [torch.rand(N, 8), torch.rand(N, 64)],  # list
        torch.rand(N, 8),  # tensor
        torch.rand(N, 8, 4, 2),  # 4-dim tensor
    ])
    # yapf: enable
    def test_num_samples_infer(self, batch: Any, dataspec: DataSpec):
        assert dataspec._default_get_num_samples_in_batch(batch) == N

    def test_batch_dict_mismatch(self, dataspec: DataSpec):
        N = 128
        batch = {'a': torch.rand(N, 8), 'b': torch.rand(N * 2, 64)}

        with pytest.raises(NotImplementedError, match='multiple Tensors'):
            dataspec._default_get_num_samples_in_batch(batch)

    def test_unable_to_infer(self, dataspec: DataSpec):
        N = 128
        batch = [torch.rand(N, 8), 'I am a string.']

        with pytest.raises(ValueError, match='Unable to determine'):
            dataspec._default_get_num_samples_in_batch(batch)
