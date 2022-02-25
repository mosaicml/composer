# Copyright 2021 MosaicML. All Rights Reserved.

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional

import yahp as hp

from composer.core.types import JSON
from composer.models.model_hparams import ModelHparams


@dataclass
class TransformerHparams(ModelHparams, ABC):
    """Defines the necessary hyparameters for a Transformer base module."""

    tokenizer_name: str = hp.optional("Tokenizer name to pull from Huggingface Model Hub.", default=None)
    pretrained_model_name: Optional[str] = hp.optional(
        doc="Pretrained model name to pull from Huggingface Model Hub.",
        default=None,
    )
    model_config: Dict[str, JSON] = hp.optional(doc="A dictionary providing a HuggingFace model configuration.",
                                                default_factory=dict)
    use_pretrained: bool = hp.optional("Whether to initialize the model with the pretrained weights.", default=False)
    gradient_checkpointing: bool = hp.optional("Whether to enable gradient checkpointing.", default=False)

    def validate(self):
        if self.tokenizer_name is None:
            raise ValueError('tokenizer_name cannot be None. Enter model name to pull from HuggingFace Model Hub.')

        if self.pretrained_model_name is None and self.model_config == {}:
            raise Exception("One of pretrained_model_name or model_config needed.")

        if self.pretrained_model_name is not None and self.model_config != {}:
            raise Exception("Only one of pretrained_model_name or model_config can be provided.")

        if self.use_pretrained and self.model_config:
            raise Exception("A model cannot load pretrained weights from configuration.")
