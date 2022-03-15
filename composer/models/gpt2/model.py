# Copyright 2021 MosaicML. All Rights Reserved.

"""GPT-2 model based on `Hugging Face GPT-2 <https://huggingface.co/docs/transformers/master/en/model_doc/gpt2>`_.

Implemented as a wrapper using :class:`.ComposerTrainer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from torchmetrics.collections import MetricCollection

from composer.models.nlp_metrics import Perplexity
from composer.models.transformer_shared import ComposerTransformer

if TYPE_CHECKING:
    import transformers

    from composer.core.types import Batch, Metrics, Tensors

__all__ = ["GPT2Model"]


class GPT2Model(ComposerTransformer):
    """Implements :class:`~composer.models.transformer_shared.ComposerTransformer` to wrap `Hugging Face GPT-2
    transformers <https://huggingface.co/docs/transformers/master/en/model_doc/gpt2#overview>`_. Logs training and
    validation perplexity.

    From `Language Models are Unsupervised Multitask Learners <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_ (Radford et al, 2018).

    Args:
        module (transformers.GPT2Model): The model to wrap with this module.
        config (transformers.GPT2Config): The config for the model.
        tokenizer (transformers.GPT2Tokenizer): The tokenizer used for this model. Necessary to process model inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. default: ``False``.

    To create a GPT-2 model for language modeling pretraining:

    .. testcode::

        from composer.models import GPT2Model
        import transformers

        config = transformers.GPT2Config()
        hf_model = transformers.GPT2LMHeadModel(config=config) # gpt2-small model from huggingface
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2Model(module=hf_model, config=config, tokenizer=tokenizer)
    """

    def __init__(self,
                 module: transformers.GPT2Model,
                 config: transformers.GPT2Config,
                 tokenizer: transformers.GPT2Tokenizer,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__(
            module=module,  #type: ignore (thirdparty)
            config=config,
            tokenizer=tokenizer,
            gradient_checkpointing=gradient_checkpointing)

        # If we ever have algorithms that modify the loss function, then this might be a bit inefficient
        #  because it'll compute the expensive softmax operation twice.
        # Instead, we should consider figuring out how to leverage self.train_loss and return the e^self.train_loss.
        # Of course, this also depends on the implementation details of algorithms.
        self.train_perplexity = Perplexity()
        self.val_perplexity = Perplexity()

    def loss(self, outputs: Mapping, batch: Batch) -> Tensors:
        if outputs.get('loss', None) is not None:
            return outputs['loss']
        else:
            raise NotImplementedError('Calculating loss directly not supported yet.')

    def metrics(self, train: bool = False) -> Metrics:
        return MetricCollection([self.train_loss, self.train_perplexity]) if train else MetricCollection(
            [self.val_loss, self.val_perplexity])
