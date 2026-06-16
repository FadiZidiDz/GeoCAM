"""Frozen DINOv2 ViT-S/14 optical encoder."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class OpticalEncoder(nn.Module):
    """Wraps DINOv2 ViT-S/14 via torch.hub; parameters are frozen."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Load backbone and optionally freeze weights.

        Args:
            cfg: Full GeoCAM config dict.
        """
        super().__init__()
        name = cfg["model"]["optical"].get("backbone", "dinov2_vits14")
        if name != "dinov2_vits14":
            raise ValueError(f"Unsupported optical backbone: {name}")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True,
        )
        frozen = bool(cfg["model"]["optical"].get("frozen", True))
        self.frozen = frozen
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True) -> "OpticalEncoder":
        """Keep DINO backbone in eval mode when frozen."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB crops to CLS features ``(B, 384)``.

        Args:
            x: Tensor ``(B, 3, H, W)`` (ImageNet-normalized).

        Returns:
            L2-ready CLS embeddings (not normalized here).
        """
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            z = out["x_norm_clstoken"]
        else:
            z = out[:, 0]
        return z

    def encode_set(self, x: torch.Tensor, batch_size: int, n: int) -> torch.Tensor:
        """Reshape flat optical batch to ``(B, N, D)``.

        Args:
            x: Tensor ``(B*N, 3, H, W)``.
            batch_size: B.
            n: Number of optical views per tile.

        Returns:
            Tensor ``(B, N, D)``.
        """
        feats = self.forward(x)
        return feats.view(batch_size, n, -1)
