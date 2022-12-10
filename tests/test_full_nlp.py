# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parameterize('model_type', ['tinybert', 'simpletransformer'])
def test_full_nlp_pipeline():
    # load random lm dataset
    # pretrain base model, checkpoint, eval
    # load random classification dataset
    # load for finetuning, checkpoint, eval
    # load for inference, eval
    pass
