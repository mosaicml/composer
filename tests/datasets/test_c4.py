# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from composer.datasets.c4 import C4Dataset


@pytest.mark.skip()
@pytest.mark.parametrize('num_samples', [500, 500000])
@pytest.mark.parametrize('group_method', ['concat', 'truncate'])
def test_c4_length(num_samples, group_method):
    # For large values of `num_samples` such as 500000, this is greater than the size of the validation set
    # and tests whether we can safely loop over the underlying dataset.
    # But it can take tens of minutes, depending on internet bandwidth, so we skip it in CI testing.

    dataset = C4Dataset(split='validation',
                        num_samples=num_samples,
                        tokenizer_name='gpt2',
                        max_seq_len=1000,
                        group_method=group_method,
                        shuffle=False,
                        seed=1)

    count = 0
    for _ in dataset:
        count += 1

    assert count == num_samples, f'Expected length {num_samples}, got length {count}'
