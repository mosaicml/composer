# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from transformers.models.bert.modeling_bert import (BertAttention, BertEmbeddings, BertIntermediate, BertOutput,
                                                    BertPreTrainedModel, BertSelfOutput)


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
        # TODO: Consider adding back in fuse_bias
        self.fuse_bias = False # getattr(config, 'fused_bias_mha', False)
        
        linear_cls = nn.Linear  # if not self.fuse_bias else FusedDenseResidual
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
