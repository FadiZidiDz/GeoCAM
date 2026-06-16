"""Set Attention Aggregator (perceiver-style) over optical embeddings."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class SetAttentionAggregator(nn.Module):
    """Aggregate a variable-size set of embeddings with a learned query."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Build SAA from ``cfg['model']['saa']``.

        Args:
            cfg: Full GeoCAM config dict.
        """
        super().__init__()
        s = cfg["model"]["saa"]
        self.embed_dim = int(s["embed_dim"])
        self.num_heads = int(s["num_heads"])  # stored so forward() can expand attn_mask
        ffn_ratio = float(s["ffn_ratio"])
        dropout = float(s.get("dropout", 0.0))
        self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden = int(self.embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        geo_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate set ``x`` into one vector per batch row.

        Args:
            x: Tensor ``(B, N, D)``.
            padding_mask: If provided, bool tensor ``(B, N)`` with True at padded slots.
            geo_weights: Optional ``(B, N)`` float overlap ratios in ``[0, 1]``.
                When provided, added as a log-scale additive bias to attention logits
                so that optical images with higher geographic overlap receive more weight.

        Returns:
            ``(agg, attn_weights)`` with shapes ``(B, D)`` and ``(B, 1, N)``.
        """
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        x_norm = self.norm1(x)
        # Build a single float attn_mask that encodes BOTH:
        # - padding (padded slots get a large negative bias)
        # - geo overlap bias (log(w))
        #
        # This avoids deprecated mismatched key_padding_mask (bool) + attn_mask (float).
        attn_mask = None
        if padding_mask is not None or geo_weights is not None:
            n = x.size(1)
            bias = torch.zeros((b, 1, n), device=x.device, dtype=x.dtype)
            if geo_weights is not None:
                bias = bias + geo_weights.to(dtype=x.dtype).clamp(min=0.1).log().unsqueeze(1)
            if padding_mask is not None:
                bias = bias.masked_fill(padding_mask.unsqueeze(1), -1e4)
            # MultiheadAttention 3D mask expects (B*num_heads, Tq, Tk).
            attn_mask = bias.repeat_interleave(self.num_heads, dim=0).contiguous()

        attn_out, attn_weights = self.cross_attn(
            q, x_norm, x_norm,
            key_padding_mask=None,
            attn_mask=attn_mask,
        )
        q = q + attn_out
        q = q + self.ffn(self.norm2(q))
        return q.squeeze(1), attn_weights

