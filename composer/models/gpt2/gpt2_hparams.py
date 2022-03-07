# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses
import textwrap
from typing import TYPE_CHECKING

from composer.models.transformer_hparams import TransformerHparams

if TYPE_CHECKING:
    from composer.models.transformer_shared import ComposerTransformer


@dataclasses.dataclass
class GPT2Hparams(TransformerHparams):
    """Overrides TransformerHparams to create GPT-2 specific models and configs."""

    def initialize_object(self) -> "ComposerTransformer":
        try:
            import transformers
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
                if using pip or `conda install -c conda-forge transformers` if using Anaconda.""")) from e
        from composer.models.gpt2.model import GPT2Model
        self.validate()

        if self.model_config:
            config = transformers.GPT2Config.from_dict(self.model_config)
        elif self.pretrained_model_name is not None:
            config = transformers.GPT2Config.from_pretrained(self.pretrained_model_name)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')

        # setup the tokenizer in the hparams interface
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(self.tokenizer_name)

        if self.use_pretrained:
            model = transformers.AutoModelForCausalLM.from_pretrained(self.pretrained_model_name)
        else:
            model = transformers.AutoModelForCausalLM.from_config(config)  #type: ignore (thirdparty)

        return GPT2Model(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer=tokenizer,
            gradient_checkpointing=self.gradient_checkpointing,
        )
