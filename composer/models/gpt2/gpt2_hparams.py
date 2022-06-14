# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`."""

from dataclasses import dataclass
from typing import Dict, Optional

import yahp as hp

from composer.core.types import JSON
from composer.models.model_hparams import ModelHparams

__all__ = ["GPT2Hparams"]


@dataclass
class GPT2Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`.

    Args:
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
    """
    model_config: Optional[Dict[str,
                                JSON]] = hp.optional(doc="A dictionary providing a HuggingFace model configuration.",
                                                     default_factory=dict)
    use_pretrained: Optional[bool] = hp.optional("Whether to initialize the model with the pretrained weights.",
                                                 default=False),
    gradient_checkpointing: Optional[bool] = hp.optional("Whether to enable gradient checkpointing.", default=False)
    def initialize_object(self):
        from composer.models.gpt2.model import create_gpt2

        # user must specify either config or the pretrained model
        if (not self.model_config and not self.use_pretrained):
            raise ValueError('One of use_pretrained or model_config needed.')

        elif (self.model_config and self.use_pretrained):
            raise ValueError('Cannot load model from both model_config and use_pretrained')

        return create_gpt2(
            model_config=self.model_config,  #type: ignore (thirdparty)
            use_pretrained=self.use_pretrained,
            gradient_checkpointing=self.gradient_checkpointing,
        )
