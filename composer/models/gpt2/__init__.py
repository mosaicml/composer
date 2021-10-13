# Copyright 2021 MosaicML. All Rights Reserved.

from composer.models.gpt2.gpt2_hparams import GPT2Hparams as GPT2Hparams
from composer.models.gpt2.model import GPT2Model as GPT2Model

_metadata = {
    'gpt2': {
        '_task': 'Language Modeling',
        '_dataset': 'OpenWebText',
        '_name': 'GPT-2 52M',
        '_quality': '30.88',
        '_metric': 'Perplexity',
        '_ttt': '02:44',
        '_hparams': 'gpt2_52m.yaml'
    },
    'gpt2 -- TODO RENAME TO GPT2': {
        '_task': 'Language Modeling',
        '_dataset': 'OpenWebText',
        '_name': 'GPT-2 83M',
        '_quality': '26.57',
        '_metric': 'Perplexity',
        '_ttt': '04:52',
        '_hparams': 'gpt2_83m.yaml'
    },
    'gpt2 --! TODO RENAME TO GPT2': {
        '_task': 'Language Modeling',
        '_dataset': 'OpenWebText',
        '_name': 'GPT-2 125M',
        '_quality': '24.04',
        '_metric': 'Perplexity',
        '_ttt': '08:25',
        '_hparams': 'gpt2_125m.yaml'
    }
}