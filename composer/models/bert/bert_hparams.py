# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ general and classification interfaces for
:class:`.BERTModel`."""

from dataclasses import dataclass
from typing import Dict, Optional

import yahp as hp

from composer.core.types import JSON
from composer.models.model_hparams import ModelHparams

__all__ = ["BERTForClassificationHparams", "BERTHparams"]


@dataclass
class BERTHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.BERTModel`.

    Args:
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. default: False.
    """
    model_config: Optional[Dict[str,
                                JSON]] = hp.optional(doc="A dictionary providing a HuggingFace model configuration.",
                                                     default_factory=dict)
    use_pretrained: Optional[bool] = hp.optional("Whether to initialize the model with the pretrained weights.",
                                                 default=False)
    gradient_checkpointing: Optional[bool] = hp.optional("Whether to enable gradient checkpointing.", default=False)

    def initialize_object(self):
        from composer.models.bert.model import create_bert_mlm

        # user must specify either config or the pretrained model
        if (not self.model_config and not self.use_pretrained):
            raise ValueError('One of use_pretrained or model_config needed.')

        elif (self.model_config and self.use_pretrained):
            raise ValueError('Cannot load model from both model_config and use_pretrained')

        return create_bert_mlm(
            model_config=self.model_config,  # type: ignore (thirdparty)
            use_pretrained=self.use_pretrained,
            gradient_checkpointing=self.gradient_checkpointing,
        )


@dataclass
class BERTForClassificationHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.BERTModel`.

    Args:
        num_labels (int, optional): The number of classes in the classification task. Default: ``2``.
        model_config (Dict[str, JSON]): A dictionary providing a HuggingFace model configuration.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. default: False.
    """
    num_labels: Optional[int] = hp.optional(doc="The number of possible labels for the task.", default=2)
    model_config: Optional[Dict[str,
                                JSON]] = hp.optional(doc="A dictionary providing a HuggingFace model configuration.",
                                                     default_factory=dict)
    use_pretrained: Optional[bool] = hp.optional("Whether to initialize the model with the pretrained weights.",
                                                 default=False)
    gradient_checkpointing: Optional[bool] = hp.optional("Whether to enable gradient checkpointing.", default=False)

    def initialize_object(self):
        from composer.models.bert.model import create_bert_classification

        # user must specify either config or the pretrained model
        if (not self.model_config and not self.use_pretrained):
            raise ValueError('One of use_pretrained or model_config needed.')

        elif (self.model_config and self.use_pretrained):
            raise ValueError('Cannot load model from both model_config and use_pretrained')

        return create_bert_classification(
            num_labels=self.num_labels,
            model_config=self.model_config,  # type: ignore (thirdparty)
            use_pretrained=self.use_pretrained,
            gradient_checkpointing=self.gradient_checkpointing,
        )
