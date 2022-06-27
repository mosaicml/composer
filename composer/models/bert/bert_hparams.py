# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ general and classification interfaces for
:class:`.BERTModel`."""

from dataclasses import dataclass
from typing import Dict, Optional

import yahp as hp

from composer.core.types import JSON
from composer.models.model_hparams import ModelHparams

__all__ = ['BERTForClassificationHparams', 'BERTHparams']


@dataclass
class BERTHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.BERTModel`.

    Args:
        model_config (Dict[str, JSON], optional): A dictionary providing a HuggingFace model configuration.
        pretrained_model_name (str, optional): Pretrained model name to pull from Hugging Face Model Hub.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights.
        tokenizer_name (str, optional): The tokenizer used for this model,
            necessary to assert required model inputs. Default ``None``.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. default: False.
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
        from composer.models.bert.model import create_bert_mlm

        # user must specify one of either config or the pretrained model
        if not self.pretrained_model_name and self.model_config == {}:
            raise Exception('One of pretrained_model_name or model_config needed.')

        if self.use_pretrained and self.model_config:
            raise Exception('A model cannot load pretrained weights from configuration.')

        return create_bert_mlm(
            model_config=self.model_config,  # type: ignore (thirdparty)
            pretrained_model_name=self.pretrained_model_name,
            use_pretrained=self.use_pretrained,
            tokenizer_name=self.tokenizer_name,
            gradient_checkpointing=self.gradient_checkpointing,
        )


@dataclass
class BERTForClassificationHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.BERTModel`.

    Args:
        num_labels (int, optional): The number of classes in the classification task. Default: ``2``.
        pretrained_model_name (str, optional): Pretrained model name to pull from Hugging Face Model Hub.
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights.
        tokenizer_name (Optional[str]): The tokenizer used for this model,
            necessary to assert required model inputs. Default ``None``.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. default: False.
    """
    num_labels: Optional[int] = hp.optional(doc='The number of possible labels for the task.', default=2)
    pretrained_model_name: Optional[str] = hp.optional(doc='Pretrained model name to pull from Hugging Face Model Hub.',
                                                       default=None)
    model_config: Optional[Dict[str,
                                JSON]] = hp.optional(doc='A dictionary providing a HuggingFace model configuration.',
                                                     default_factory=dict)
    use_pretrained: Optional[bool] = hp.optional('Whether to initialize the model with the pretrained weights.',
                                                 default=False)
    tokenizer_name: Optional[str] = hp.optional(
        'The tokenizer used for this model, necessary to assert required model inputs.', default=None)
    gradient_checkpointing: Optional[bool] = hp.optional('Whether to enable gradient checkpointing.', default=False)

    def initialize_object(self):
        from composer.models.bert.model import create_bert_classification

        # user must specify one of either config or the pretrained model
        if not self.pretrained_model_name and self.model_config == {}:
            raise Exception('One of pretrained_model_name or model_config needed.')

        if self.use_pretrained and self.model_config:
            raise Exception('A model cannot load pretrained weights from configuration.')

        return create_bert_classification(
            num_labels=self.num_labels,
            pretrained_model_name=self.pretrained_model_name,
            model_config=self.model_config,  # type: ignore (thirdparty)
            use_pretrained=self.use_pretrained,
            tokenizer_name=self.tokenizer_name,
            gradient_checkpointing=self.gradient_checkpointing,
        )
