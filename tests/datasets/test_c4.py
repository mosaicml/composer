import pytest

from composer.datasets.c4 import C4Dataset


@pytest.mark.timeout(30)
@pytest.mark.parametrize("num_samples", [500, 500000])
@pytest.mark.parametrize("group_method", ["concat", "truncate"])
def test_c4_length(num_samples, group_method):
    if num_samples == 500000:
        # This is greater than the size of the validation set,
        # and tests whether we can safely loop over the underlying dataset when `num_samples` is large.
        # But it can take tens of minutes, depending on internet bandwidth, so we skip it in CI testing.
        pytest.skip()

    dataset = C4Dataset(split="validation",
                        num_samples=num_samples,
                        tokenizer_name="gpt2",
                        max_seq_len=1000,
                        group_method=group_method,
                        shuffle=False,
                        seed=1)

    count = 0
    for _ in dataset:
        count += 1

    assert count == num_samples, f"Expected length {num_samples}, got length {count}"
