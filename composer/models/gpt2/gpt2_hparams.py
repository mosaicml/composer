# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`."""

import dataclasses
from typing import TYPE_CHECKING

from composer.models.transformer_hparams import TransformerHparams
from composer.utils.import_helpers import MissingConditionalImportError

if TYPE_CHECKING:
    from composer.models.transformer_shared import ComposerTransformer

__all__ = ["GPT2Hparams"]


@dataclasses.dataclass
class GPT2Hparams(TransformerHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.GPT2Model`.

    Args:
        pretrained_model_name (str): Pretrained model name to pull from Hugging Face Model Hub.
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        tokenizer_name (Optional[str]): The tokenizer used for this model,
            necessary to assert required model inputs.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
    """

    def initialize_object(self) -> "ComposerTransformer":
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group="nlp", conda_package="transformers") from e

        from composer.models.gpt2.model import GPT2Model
        self.validate()

        if self.model_config:
            config = transformers.GPT2Config.from_dict(self.model_config)
        elif self.pretrained_model_name is not None:
            # TODO (Moin): verify that the config is an appropriate instance of GPT2!
            config = transformers.GPT2Config.from_pretrained(self.pretrained_model_name)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')

        # setup the tokenizer in the hparams interface
        if self.tokenizer_name is not None:
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(self.tokenizer_name)
        else:
            tokenizer = None

        if self.use_pretrained:
            assert transformers.AutoModelForCausalLM.from_pretrained is not None, "from_pretrained should not be None"
            model = transformers.AutoModelForCausalLM.from_pretrained(self.pretrained_model_name)
        else:
            model = transformers.AutoModelForCausalLM.from_config(config)  #type: ignore (thirdparty)

        return GPT2Model(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer=tokenizer,
            gradient_checkpointing=self.gradient_checkpointing,
        )
