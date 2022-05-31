# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

#import pdb

import pytest
import torch

from tests.fixtures import synthetic_hf_state


@pytest.mark.parametrize("config", synthetic_hf_state.make_dataset_configs())
def test_synthetic_hf_state(config, synthetic_hf_state_fixture):
    state = synthetic_hf_state_fixture
    lm, dataloader = synthetic_hf_state.model_components(config)
    sample = next(iter(dataloader)).data
    state.batch = next(iter(state.dataloader)).data
    assert state.batch.keys() == sample.keys()
    for key in state.batch.keys():
        assert state.batch[key].size() == sample[key].size()
    lm.eval()
    logits, labels = lm.validate(sample)
    assert hasattr(state, "batch")
    state_output = state.model(state.batch)
    if labels is not None:
        assert isinstance(logits, torch.Tensor)
        assert state_output['logits'].size() == logits.size()
        assert state.batch['labels'].size() == labels.size()
    else:
        assert state_output['logits'].size() == logits['logits'].size()
