"""Patch-wise MSE with per-patch MAE-style normalization."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchReconstructionLoss(nn.Module):
    """MSE on masked patches after per-patch mean/variance normalization."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        """Args:
            eps: Numerical stability for variance.
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
        masked_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized MSE on masked locations.

        Args:
            pred_patches: Predictions ``(B, n_mask, patch_dim)``.
            target_patches: Full patch tensor ``(B, L, patch_dim)``.
            masked_ids: Indices ``(B, n_mask)`` into ``L`` for masked patches.

        Returns:
            Scalar MSE loss.
        """
        expanded = masked_ids.unsqueeze(-1).expand_as(pred_patches)
        target = torch.gather(target_patches, 1, expanded)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, unbiased=False, keepdim=True)
        target_norm = (target - mean) / torch.sqrt(var + self.eps)
        return torch.mean((pred_patches - target_norm) ** 2)
