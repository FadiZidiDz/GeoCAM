"""L2-normalized MLP projection heads for contrastive embeddings."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP followed by L2 normalization."""

    def __init__(self, cfg: Dict[str, Any], in_dim: int) -> None:
        """Build head using ``cfg['model']['projection_head']``.

        Args:
            cfg: Full GeoCAM config dict.
            in_dim: Input feature dimension (e.g. 384).
        """
        super().__init__()
        p = cfg["model"]["projection_head"]
        hidden = int(p["hidden_dim"])
        out_dim = int(p["out_dim"])
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Args:
            x: Tensor ``(B, in_dim)``.

        Returns:
            Tensor ``(B, out_dim)`` with unit row norms.
        """
        x = self.net(x)
        return F.normalize(x, dim=-1)
