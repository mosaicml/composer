# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.filterwarnings(
    r'ignore:Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer:UserWarning:torchmetrics')
def test_synthetic_hf_state(synthetic_hf_state):
    pytest.importorskip('transformers')

    state, lm, dataloader = synthetic_hf_state
    sample = next(iter(dataloader)).data
    state.batch = next(iter(state.dataloader)).data
    assert state.batch.keys() == sample.keys()
    for key in state.batch.keys():
        assert state.batch[key].size() == sample[key].size()
    lm.eval()
    logits, labels = lm.validate(sample)
    assert hasattr(state, 'batch')
    state_output = state.model(state.batch)
    if labels is not None:
        assert isinstance(logits, torch.Tensor)
        assert state_output['logits'].size() == logits.size()
        assert state.batch['labels'].size() == labels.size()
    else:
        assert state_output['logits'].size() == logits['logits'].size()
