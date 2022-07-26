# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def test_hf_model_forward():
    pytest.importorskip('transformers')
    import transformers
    from transformers.modeling_outputs import SequenceClassifierOutput

    from composer.models import HuggingFaceModel

    # dummy sequence batch with 2 labels, 32 sequence length, and 30522 (bert) vocab size).
    input_ids = torch.randint(low=0, high=30522, size=(2, 32))
    labels = torch.randint(low=0, high=1, size=(2,))
    token_type_ids = torch.zeros(size=(2, 32), dtype=torch.int64)
    attention_mask = torch.randint(low=0, high=1, size=(2, 32))
    batch = {
        'input_ids': input_ids,
        'labels': labels,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }

    # non pretrained model to avoid a slow test that downloads the weights.
    config = transformers.AutoConfig.from_pretrained('bert-base-uncased', num_labels=2)
    hf_model = transformers.AutoModelForSequenceClassification.from_config(config)  # type: ignore (thirdparty)
    model = HuggingFaceModel(hf_model)

    out = model(batch)
    assert isinstance(out, SequenceClassifierOutput)
    assert out.logits.shape == (2, 2)
