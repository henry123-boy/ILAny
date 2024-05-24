'''
modified from
https://github.com/zju3dv/LoFTR/blob/master/src/loftr/loftr_module/transformer.py
'''
import torch
from torch.nn import Module, Dropout
import copy
import torch.nn as nn
import torch.nn.functional as F

class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        # QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        # if kv_mask is not None:
        #     QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float(-1e12))
        # softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        # A = torch.softmax(softmax_temp * QK, dim=2)
        # if self.use_dropout:
        #     A = self.dropout(A)
        # queried_values_ = torch.einsum("nlsh,nshd->nlhd", A, values)

        # Compute the attention and the weighted average
        input_args = [x.contiguous() for x in [queries.permute(0,2,1,3), keys.permute(0,2,1,3), values.permute(0,2,1,3)]]
        queried_values = F.scaled_dot_product_attention(*input_args).permute(0,2,1,3).float()  # type: ignore


        return queried_values.contiguous()

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,):
        super(TransformerEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message