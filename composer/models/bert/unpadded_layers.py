# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import (BertAttention, BertEmbeddings, BertIntermediate,
                                                    BertOutput, BertPredictionHeadTransform, BertPreTrainedModel,
                                                    BertSelfOutput)

from composer.models.bert.ops.fused_dense import FusedDenseResidual  # FusedDenseTD
from composer.models.bert.ops.fused_dense import fused_dense_function_td


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = input.shape[0]
        assert input.ndim == 2
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(input, 0, repeat(indices, 'z -> z d', d=input.shape[1]))

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_input = torch.zeros([ctx.first_axis_dim, *grad_output.shape[1:]],
                                 device=grad_output.device,
                                 dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, 'z -> z d', d=grad_output.shape[1]), grad_output)
        return grad_input, None


index_first_axis = IndexFirstAxis.apply


class BertFlashSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention '
                             f'heads ({config.num_attention_heads})')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.p_dropout = config.attention_probs_dropout_prob

        self.fuse_bias = getattr(config, 'fused_bias_mha', False)

        linear_cls = nn.Linear if not self.fuse_bias else FusedDenseResidual
        self.Wqkv = linear_cls(self.all_head_size, 3 * config.hidden_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen_in_batch):
        """
        Arguments:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,), torch.int32
            max_seqlen_in_batch: int
        Return:
            context: (total_nnz, dim)
        """
        if not self.fuse_bias:
            qkv = self.Wqkv(hidden_states)
        else:
            qkv, hidden_states = self.Wqkv(hidden_states)  # (total_nnz, 3 * dim)
        qkv = rearrange(qkv, 'nnz (t h d) -> nnz t h d', t=3, h=self.num_attention_heads)
        orig_dtype = qkv.dtype
        qkv = qkv.to(torch.float16)
        context = flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen_in_batch,
                                                     self.p_dropout if self.training else 0.0)
        context = context.to(orig_dtype)
        return rearrange(context, 'nnz h d -> nnz (h d)'), hidden_states


class BertFlashAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertFlashSelfAttention(config)
        # TODO: Use custom BertSelfOutput instead of HuggingFace version
        # to use FusedDenseTD
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, cu_seqlens, max_s, subset_idx=None):
        """subset_idx: set of indices whose values we care about at the end of the layer
        (e.g., the masked tokens, if this is the final layer).
        """
        self_output, input_tensor = self.self(input_tensor, cu_seqlens, max_s)
        if subset_idx is not None:
            return self.output(index_first_axis(self_output, subset_idx), index_first_axis(input_tensor, subset_idx))
        else:
            return self.output(self_output, input_tensor)


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.flash_attn = config.unpad
        if config.unpad_flash_attn:
            self.attention = BertFlashAttention(config)
        else:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    # def forward(self, hidden_states, attention_mask, seqlen, batch):
    def forward(self, hidden_states, attention_mask, seqlen, subset_idx=None):
        """subset_idx: set of indices whose values we care about at the end of the layer
        (e.g., the masked tokens, if this is the final layer).
        """
        if self.flash_attn:
            attention_output = self.attention(hidden_states, attention_mask, seqlen, subset_idx)
        else:
            attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.num_attention_heads = config.num_attention_heads
        self.unpad = config.unpad
        self.unpad_flash_attn = config.unpad_flash_attn

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, subset_mask=None):
        all_encoder_layers = []
        batch = None
        seqlen = None
        if self.unpad_flash_attn:
            # attention_mask_bool = rearrange(attention_mask, 'b 1 1 s -> b s') == 0.0
            attention_mask_bool = attention_mask.bool()
            batch, seqlen = hidden_states.shape[:2]
            # Unpad inputs and mask. It will remove tokens that are padded. Assume ntokens is total number of tokens (padded and non-padded)
            # and ntokens_unpad is total number of non-padded tokens. Then unpadding performs the following compression of the inputs:
            #        hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask_bool)
            if subset_mask is None:
                for layer_module in self.layer:
                    hidden_states = layer_module(hidden_states, cu_seqlens, max_seqlen_in_batch)
                    if output_all_encoded_layers:
                        all_encoder_layers.append(hidden_states)
                # Pad inputs and mask. It will insert back zero-padded tokens. Assume ntokens is total number of tokens (padded and non-padded)
                # and ntokens_unpad is total number of non-padded tokens. Then padding performs the following de-compression:
                #        hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            else:
                for i in range(len(self.layer) - 1):
                    layer_module = self.layer[i]
                    hidden_states = layer_module(hidden_states, cu_seqlens, max_seqlen_in_batch)
                    if output_all_encoded_layers:
                        all_encoder_layers.append(hidden_states)
                subset_idx = torch.nonzero(subset_mask[attention_mask_bool], as_tuple=False).flatten()
                hidden_states = self.layer[-1](hidden_states, cu_seqlens, max_seqlen_in_batch, subset_idx=subset_idx)
        else:
            raise RuntimeError('Please set unpad_flash_attention to True to use Bert unpadded model.')

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        #self.pooler = BertPooler(config)
        self.pooler = None
        self.post_init()
        #self.apply(self.init_weights)
        self.unpad = config.unpad

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                output_all_encoded_layers=False,
                masked_tokens_mask=None,
                **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask  #.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        if self.unpad == False:
            extended_attention_mask = extended_attention_mask.to(dtype=next(
                self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       output_all_encoded_layers=output_all_encoded_layers,
                                       subset_mask=subset_mask)

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][masked_tokens_mask[attention_mask_bool][subset_idx]]
            pool_input = encoder_outputs[-1][first_col_mask[attention_mask_bool][subset_idx]]
            pooled_output = (self.pooler(pool_input, pool=False) if self.pooler is not None else None)

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output
        return encoder_outputs, pooled_output


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.fused_fc = getattr(config, 'fused_bias_fc_loss_head', False)
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        if not self.fused_fc:
            hidden_states = self.decoder(hidden_states) + self.bias
        else:
            hidden_states = fused_dense_function_td(hidden_states, self.decoder.weight, self.bias)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


#####################
# Model class
#####################
class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            warnings.warn('If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for '
                          'bi-directional self-attention.')

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)

        # if dense_seq_output, prediction scores are only computed for masked tokens,
        # and the (bs, seqlen) dimensions are flattened
        self.dense_seq_output = getattr(config, 'dense_seq_output', False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError('The PAD token should be defined for generation')

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full((effective_batch_size, 1),
                                 self.config.pad_token_id,
                                 dtype=torch.long,
                                 device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}
