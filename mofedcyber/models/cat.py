"""Cross-attention Transformer (CAT) detector, Section IV-A and Fig. 2 of the paper.

Each flow is split into three views (protocol, temporal, payload). Every feature
becomes a token: value * w_j + b_j (a learned per-feature embedding), followed by a
two-layer perceptron shared within the view. Three blocks attend ACROSS views:

    block 1: queries = protocol tokens, keys/values = temporal tokens
    block 2: queries = temporal tokens, keys/values = payload tokens
    block 3: queries = payload tokens,  keys/values = protocol tokens

Each block = cross-attention -> self-attention -> feed-forward, each sublayer with a
residual connection and layer normalization. The three outputs are mean-pooled,
combined by gated fusion h = sigmoid(W_g [h_p; h_t; h_s]) * W_f [h_p; h_t; h_s],
and classified by a two-layer head. With the default sizes (d_e = 256, 8 heads,
FFN 1280, head 1280) the model has about 4.8M parameters.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

VIEWS = ("protocol", "temporal", "payload")


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, dim) * dim**-0.5)
        self.bias = nn.Parameter(torch.zeros(num_features, dim))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F_v)
        tok = x.unsqueeze(-1) * self.weight + self.bias  # (B, F_v, d)
        return self.mlp(tok)


class CrossViewBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, dim)
        )
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, need_weights: bool = False):
        a, w = self.cross(q, kv, kv, need_weights=need_weights, average_attn_weights=False)
        h = self.n1(q + self.drop(a))
        s, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.n2(h + self.drop(s))
        h = self.n3(h + self.drop(self.ffn(h)))
        return h, w  # w: (B, heads, Lq, Lk)


class CAT(nn.Module):
    def __init__(
        self,
        view_index: Dict[str, List[int]],
        num_classes: int,
        embed_dim: int = 256,
        num_blocks: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 1280,
        head_hidden: int = 1280,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_blocks != 3:
            raise ValueError("CAT uses exactly one block per view pair (num_blocks=3).")
        self.view_index = {v: list(view_index[v]) for v in VIEWS}
        for v in VIEWS:
            self.register_buffer(
                f"idx_{v}", torch.tensor(self.view_index[v], dtype=torch.long), persistent=False
            )
        self.tokenizers = nn.ModuleDict(
            {v: FeatureTokenizer(len(self.view_index[v]), embed_dim) for v in VIEWS}
        )
        self.pairs = [("protocol", "temporal"), ("temporal", "payload"), ("payload", "protocol")]
        self.blocks = nn.ModuleList(
            [CrossViewBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in self.pairs]
        )
        self.gate = nn.Linear(3 * embed_dim, embed_dim)
        self.fuse = nn.Linear(3 * embed_dim, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )
        self.last_attention: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        tokens = {v: self.tokenizers[v](x.index_select(1, getattr(self, f"idx_{v}"))) for v in VIEWS}
        pooled, att = [], {}
        for (qv, kv), blk in zip(self.pairs, self.blocks):
            h, w = blk(tokens[qv], tokens[kv], need_weights=return_attention)
            pooled.append(h.mean(dim=1))
            if return_attention:
                att[f"{qv}->{kv}"] = w.detach()
        z = torch.cat(pooled, dim=-1)
        h = torch.sigmoid(self.gate(z)) * self.fuse(z)
        if return_attention:
            self.last_attention = att
        return self.head(h)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    def n(m):
        return sum(p.numel() for p in m.parameters())

    if not isinstance(model, CAT):
        return {"total": n(model)}
    return {
        "embeddings": n(model.tokenizers),
        "blocks": n(model.blocks),
        "fusion_head": n(model.gate) + n(model.fuse) + n(model.head),
        "total": n(model),
    }


def build_model(view_index, num_classes: int, cfg) -> CAT:
    return CAT(
        view_index,
        num_classes,
        embed_dim=cfg.embed_dim,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
    )
