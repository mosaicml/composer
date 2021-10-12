# Copyright 2021 MosaicML. All Rights Reserved.

import torch


def _attn(self, query, key, value, attention_mask=None, head_mask=None):
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


def enlarge_mask(module: torch.nn.Module, max_sequence_length: int):
    old_mask = module.bias
    new_mask = torch.tril(
        torch.ones((max_sequence_length, max_sequence_length), dtype=torch.uint8,
                   device=old_mask.device)).view(1, 1, max_sequence_length, max_sequence_length)  # type: ignore
    setattr(module, "bias", new_mask)
    return module
