# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pdb

import pytest

from composer.models.bert import (BertForClassification, BERTForClassificationHparams, BertForPretraining,
                                  BertForPretrainingHparams, BertForRegression, BERTForRegressionHparams)
from tests.common.models import generate_dummy_model_config
from tests.fixtures.synthetic_hf_state import make_dataset_configs, make_lm_tokenizer

dataset_configs = make_dataset_configs(model_family=('bert'))[0]

tokenizer = make_lm_tokenizer(dataset_configs)

for model_type in [BertForPretrainingHparams, BERTForClassificationHparams, BERTForRegressionHparams]:
    config = generate_dummy_model_config(model_type, tokenizer)
    config['max_position_embeddings'] = dataset_configs['chars_per_sample']
    model = model_type(model_config=config).initialize_object()
    pdb.set_trace()
