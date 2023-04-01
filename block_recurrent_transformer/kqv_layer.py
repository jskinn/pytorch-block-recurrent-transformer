# Copyright 2023 Google, John Skinner
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import logging
import torch
import torch.nn as nn
from nn_components import tiled_dropout


class KQVLayer(nn.Module):
    """
    Layer that calculates Query, Key and Value tensors for attention.
    Based on a class of the same name in transformer_base, see
    https://github.com/google-research/meliad/blob/main/transformer/transformer_base.py
    """

    def __init__(
        self,
        in_features: int,
        embedding_dim: int,
        value_dim: int,
        num_heads: int,
        head_size: int,
        num_positional_embeddings: int,
        compute_queries: bool = True,
        cross_attention: bool = False,
        normalise_keys: bool = True,
        pre_attention_dropout: bool = True,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.normalise_keys = bool(normalise_keys)
        self.pre_attn_dropout = bool(pre_attention_dropout)
        self.queries_layer = None
        self.queries_layer_2 = None
        self.attention_scale = None
        self.position_embeddings = None

        # LayerNorm before attention
        self.pre_attention_layernorm = nn.LayerNorm(self.num_heads * self.head_size)

        # Project to keys,values,queries
        # Disable bias.  This prevents a failure mode whereby the attention matrix
        # can become filled with very large uniform values, due to high bias.
        self.keys_layer = nn.Linear(in_features, self.num_heads * self.head_size, bias=False)
        self.values_layer = nn.Linear(in_features, self.num_heads * self.head_size, bias=False)
        if compute_queries:
            self.queries_layer = nn.Linear(in_features, self.num_heads * self.head_size, bias=False)
        if cross_attention:
            self.queries_layer_2 = nn.Linear(in_features, self.num_heads * self.head_size, bias=False)

        # When normalizing keys and queries, attention must be scaled with
        # learned parameters.
        if self.normalise_keys:
            self.attention_scale = nn.Parameter(torch.ones(self.num_heads), requires_grad=True)

        # Learned position embeddings for absolute positions.
        if num_positional_embeddings > 0:
            self.position_embeddings = nn.Parameter(torch.empty(num_positional_embeddings, embedding_dim))

    def init_weights(self):
        pass

    @staticmethod
    def _normalise_kq(key_or_query: torch.Tensor) -> torch.Tensor:
        sum_squares = key_or_query.square().sum(dim=-1, keepdim=True)
        return key_or_query * torch.rsqrt(sum_squares + 1e-6)

    def forward(self, xs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_keys, embedding_size = xs.shape
        drop_tile_shape = (1, 128, embedding_size)

        # Apply layernorm to input, rather than the output.
        # This provides better gradients through the resnet, and also avoids
        # the need for a prolonged warmup phase (https://arxiv.org/abs/2002.04745)

        # Layernorm for self-attention.
        logging.getLogger(__name__).debug("kvq: pre_attn xs = %r", xs)
        xs = torch.as_tensor(xs, dtype=self.dtype)
        xs = self.pre_attn_layernorm(xs)

        # Add (optional) learned position embeddings.
        if self.position_embeddings:
            assert xs.ndim == 3  # (b, sequence_length, embedding_size)
            assert xs.shape[-2] == self.position_embeddings.shape[-2]
            logging.getLogger(__name__).debug("kvq: learned positions.")
            xs = xs + self.position_embeddings.unsqueeze(0)

        # Pre-attention dropout.
        if self.pre_attn_dropout:
            logging.getLogger(__name__).debug("kvq: pre_attn dropout.")
            xs = tiled_dropout(xs, drop_tile_shape, self.dropout_rate, deterministic=deterministic)

        # Compute keys and values.
        keys = self.keys_layer(xs)  # (b, num_keys, num_heads * head_size)
        values = self.values_layer(xs)

        # Compute queries and cross-attention queries if necessary.
        if self.queries_layer:
            queries = self.queries_layer(xs)  # (b, num_keys, n_heads * head_size)
            logging.getLogger(__name__).debug("kvq: queries = %r", queries)
        else:
            queries = None
        if self.queries2_layer:
            queries2 = self.queries2_layer(xs)  # (b, num_keys, n_heads * head_size)
            logging.getLogger(__name__).debug("kvq: queries2 = %r", queries2)
        else:
            queries2 = None

        # Reshape to split num_heads, head_size into separate dimensions.
        kv_shape = (batch_size, num_keys, self.num_heads, self.head_size)
        keys = torch.reshape(keys, kv_shape)
        values = torch.reshape(values, kv_shape)
        if queries is not None:
            queries = torch.reshape(queries, kv_shape)
        if queries2 is not None:
            queries2 = torch.reshape(queries2, kv_shape)

        if self.normalize_keys:
            # Normalize both keys and queries.
            # The learned attention_scale_factors() will return non-None.
            logging.getLogger(__name__).debug("kvq: normalize keys, queries.")
            keys = self._normalize_kq(keys)
            if queries is not None:
                queries = self._normalize_kq(queries)
            if queries2 is not None:
                queries2 = self._normalize_kq(queries2)
        else:
            # Scale queries by 1 / sqrt(d) when using unnormalized keys,queries.
            d_scale = torch.rsqrt(torch.tensor(self.head_size, dtype=self.dtype))
            logging.getLogger(__name__).debug("kvq: scale queries by 1/sqrt(d).")
            if queries is not None:
                queries = queries * d_scale
            if queries2 is not None:
                queries2 = queries2 * d_scale

        # Return keys, values, and queries.
        return keys, values, queries, queries2
