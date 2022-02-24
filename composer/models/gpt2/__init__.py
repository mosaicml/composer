# Copyright 2021 MosaicML. All Rights Reserved.

"""The GPT-2 model family is set of transformer-based networks for autoregressive language modeling at various scales.
This family was originally proposed by OpenAI, and is trained on the OpenWebText dataset. It is useful for downstream
language generation tasks, such as summarization, translation, and dialog.

See the :doc:`Model Card </model_cards/GPT2>` for more details.
"""

from composer.models.gpt2.gpt2_hparams import GPT2Hparams as GPT2Hparams
from composer.models.gpt2.model import GPT2Model as GPT2Model

__all__ = ["GPT2Model", "GPT2Hparams"]

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
