# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the safer checkpoint format."""

from pathlib import Path

import numpy as np
import torch

from composer.utils.safe_checkpoint import SafeState


class TestSafeState:
    """Tests for the safer checkpoint format."""

    def test_convert_between_basic_data(self, tmp_path: Path) -> None:
        data = {'foo': 123, 'bar': {'a': [True, False, None], 'b': 1.23}}

        safe_state = SafeState.from_data(data)

        loaded_data = safe_state.to_data()
        assert data == loaded_data

        safe_state.save(tmp_path / 'state.safe')
        loaded_state = safe_state.load(tmp_path / 'state.safe')

        loaded_data = loaded_state.to_data()
        assert data == loaded_data

    def test_convert_with_tensor(self, tmp_path: Path) -> None:
        data = {'foo': 123, 'bar': torch.tensor([1, 2, 3]), 'baz': np.array([1, 2], dtype=np.uint8)}

        safe_state = SafeState.from_data(data)

        loaded_data = safe_state.to_data()
        assert set(loaded_data.keys()) == {'foo', 'bar', 'baz'}
        assert loaded_data['foo'] == 123
        assert torch.all(torch.tensor([1, 2, 3]).eq(loaded_data['bar']))
        assert np.array_equal(loaded_data['baz'], [1, 2])
        assert loaded_data['baz'].dtype == np.uint8

        safe_state.save(tmp_path / 'state.safe')
        loaded_state = safe_state.load(tmp_path / 'state.safe')

        loaded_data = loaded_state.to_data()
        assert set(loaded_data.keys()) == {'foo', 'bar', 'baz'}
        assert loaded_data['foo'] == 123
        assert torch.all(torch.tensor([1, 2, 3]).eq(loaded_data['bar']))
        assert np.array_equal(loaded_data['baz'], [1, 2])
        assert loaded_data['baz'].dtype == np.uint8

    def test_convert_with_multiple_tensors(self, tmp_path: Path) -> None:
        data = {
            'foo': 123,
            'bar': torch.tensor([1, 2, 3]),
            'baz': [torch.tensor([1]), torch.tensor([1.1]), np.array([1.2])],
        }

        safe_state = SafeState.from_data(data)

        loaded_data = safe_state.to_data()
        assert set(loaded_data.keys()) == {'foo', 'bar', 'baz'}
        assert loaded_data['foo'] == 123
        assert torch.all(torch.tensor([1, 2, 3]).eq(loaded_data['bar']))
        assert len(loaded_data['baz']) == 3
        assert torch.all(torch.tensor([1]).eq(loaded_data['baz'][0]))
        assert torch.all(torch.tensor([1.1]).eq(loaded_data['baz'][1]))
        assert np.array_equal(loaded_data['baz'][2], [1.2])

        safe_state.save(tmp_path / 'state.safe')
        loaded_state = safe_state.load(tmp_path / 'state.safe')

        loaded_data = loaded_state.to_data()
        assert set(loaded_data.keys()) == {'foo', 'bar', 'baz'}
        assert loaded_data['foo'] == 123
        assert len(loaded_data['baz']) == 3
        assert torch.all(torch.tensor([1]).eq(loaded_data['baz'][0]))
        assert torch.all(torch.tensor([1.1]).eq(loaded_data['baz'][1]))
        assert np.array_equal(loaded_data['baz'][2], [1.2])
