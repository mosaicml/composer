# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`."""

from dataclasses import dataclass
from typing import Dict, Optional

import yahp as hp

from composer.core.types import JSON
from composer.models.model_hparams import ModelHparams

__all__ = ['GPT2Hparams']


@dataclass
class GPT2Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`.

    Args:
        model_config (Dict[str, JSON], optional ): A dictionary providing a HuggingFace model configuration.
        pretrained_model_name (str, optional): Pretrained model name to pull from Hugging Face Model Hub.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        tokenizer_name (str, optional): The tokenizer used for this model,
            necessary to assert required model inputs. Default ``None``.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
    """
    model_config: Optional[Dict[str,
                                JSON]] = hp.optional(doc='A dictionary providing a HuggingFace model configuration.',
                                                     default_factory=dict)
    pretrained_model_name: Optional[str] = hp.optional(doc='Pretrained model name to pull from Hugging Face Model Hub.',
                                                       default=None)
    use_pretrained: Optional[bool] = hp.optional('Whether to initialize the model with the pretrained weights.',
                                                 default=False)
    tokenizer_name: Optional[str] = hp.optional(
        'The tokenizer used for this model, necessary to assert required model inputs.', default=None)
    gradient_checkpointing: Optional[bool] = hp.optional('Whether to enable gradient checkpointing.', default=False)

    def initialize_object(self):
        from composer.models.gpt2.model import create_gpt2

        # user must specify one of either config or the pretrained model
        if not self.pretrained_model_name and self.model_config == {}:
            raise Exception('One of pretrained_model_name or model_config needed.')

        if self.use_pretrained and self.model_config:
            raise Exception('A model cannot load pretrained weights from configuration.')

        return create_gpt2(
            model_config=self.model_config,  #type: ignore (thirdparty)
            pretrained_model_name=self.pretrained_model_name,
            use_pretrained=self.use_pretrained,
            tokenizer_name=self.tokenizer_name,
            gradient_checkpointing=self.gradient_checkpointing,
        )
