# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The GPT-2 model family is set of transformer-based networks for autoregressive language modeling at various scales.
This family was originally proposed by OpenAI, and is trained on the OpenWebText dataset. It is useful for downstream
language generation tasks, such as summarization, translation, and dialog.

See the :doc:`Model Card </model_cards/GPT2>` for more details.
"""

from composer.models.gpt2.model import create_gpt2 as create_gpt2

__all__ = ['create_gpt2']

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
