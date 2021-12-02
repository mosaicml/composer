# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yahp as hp

from composer.models.transformer_hparams import TransformerHparams
from composer.utils.string_enum import StringEnum

if TYPE_CHECKING:
    from composer.models.transformer_shared import MosaicTransformer


class MLMTasks(StringEnum):
    MASKED_LM = "masked_lm"
    SEQUENCE_CLASSIFICATION = "sequence_classification"


@dataclass
class BERTHparams(TransformerHparams):
    task: MLMTasks = hp.optional(doc="The training task to use (must be one of MLMTasks).", default=None)

    def initialize_object(self) -> "MosaicTransformer":
        import transformers

        from composer.models.bert.model import BERTModel
        self.validate()

        self.task_classes = {
            MLMTasks.MASKED_LM: transformers.AutoModelForMaskedLM,
            MLMTasks.SEQUENCE_CLASSIFICATION: transformers.AutoModelForSequenceClassification
        }
        self.task_class = self.task_classes[self.task]
        print(self.task_class)

        if self.model_config:
            config = transformers.BertConfig.from_dict(self.model_config)
        elif self.pretrained_model_name is not None:
            config = transformers.BertConfig.from_pretrained(self.pretrained_model_name)
        else:
            raise ValueError('One of pretrained_model_name or model_config needed.')

        if self.use_pretrained:
            # TODO (Moin): handle the warnings on not using the seq_relationship head
            model = self.task_class.from_pretrained(self.pretrained_model_name)
        else:
            model = self.task_class.from_config(config)  #type: ignore (thirdparty)

        return BERTModel(
            module=model,
            config=config,  #type: ignore (thirdparty)
            tokenizer_name=self.tokenizer_name,
        )
