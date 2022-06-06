# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`."""

from dataclasses import dataclass
from typing import Dict

import yahp as hp

from composer.models import ModelHparams
from composer.core.types import JSON


__all__ = ["GPT2Hparams"]


@dataclass
class GPT2Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`.

    Args:
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
    """
    model_config: Dict[str, JSON] = hp.optional(doc="A dictionary providing a HuggingFace model configuration.", default_factory=dict)
    use_pretrained: bool = hp.optional("Whether to initialize the model with the pretrained weights.", default=False)
    gradient_checkpointing: bool = hp.optional("Whether to enable gradient checkpointing.", default=False)

    def initialize_object(self):
        from composer.models.gpt2.model import create_gpt2
        return create_gpt2(
            model_config=self.model_config,  #type: ignore (thirdparty)
            use_pretrained=self.use_pretrained,
            gradient_checkpointing=self.gradient_checkpointing,
        )
