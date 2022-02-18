# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Tuple

import torch


def _attn(self, query, key, value, attention_mask=None, head_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replication of identically-named attention function function ("_attn") in Composer/HuggingFace GPT2 model's
    GPT2Attention (:func:`transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn`; `GitHub link <https://\\
    github.com/huggingface/transformers/blob/2e11a043374a6229ec129a4765ee4ba7517832b9/src/transformers/models/\\
    gpt2/modeling_gpt2.py#L192>`_), but this function implements ALiBi and will be used to replace the default attention
    function."""
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1))**0.5)

    # This is the modification from the original attention
    n_tokens = attn_weights.shape[-1]
    # Truncate alibi distance weights to size of current batch
    alibi = self.alibi[:, :, 0:n_tokens]
    # alibi = self.alibi[:, :, :, 0:n_tokens].repeat(batch_size, 1, 1, 1)
    attn_weights = attn_weights + alibi
    # End modification

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.Softmax(dim=-1)(attn_weights)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def enlarge_mask(module: torch.nn.Module, max_sequence_length: int) -> torch.nn.Module:
    """Increases the size of the attention mask in Composer/HuggingFace GPT2 model's GPT2Attention
    (:func:`transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn`; `GitHub link <https://\\
    github.com/huggingface/transformers/blob/2e11a043374a6229ec129a4765ee4ba7517832b9/src/transformers/\\
    models/gpt2/modeling_gpt2.py#L140>`_).

    This is necessary for evaluating on sequence lengths longer than the model was initialized to accommodate.
    """
    old_mask = module.bias
    new_mask = torch.tril(
        torch.ones(
            (max_sequence_length, max_sequence_length),  # type: ignore
            dtype=torch.uint8,
            device=old_mask.device)).view(1, 1, max_sequence_length, max_sequence_length)  # type: ignore
    setattr(module, "bias", new_mask)
    return module
