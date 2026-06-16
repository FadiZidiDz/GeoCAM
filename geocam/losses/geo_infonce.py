"""Geometry-weighted symmetric InfoNCE between sonar and aggregated optical embeddings."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoWeightedInfoNCE(nn.Module):
    """InfoNCE with per-sample weighting on positive log-probabilities."""

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negatives: bool = False,
        hard_negative_boost: float = 0.0,
    ) -> None:
        """Args:
            temperature: Softmax temperature ``tau``.
            hard_negatives: If True, upweight hard negatives in the denominator.
            hard_negative_boost: Additive boost to the hardest negative logit per row
                (applied in both s2o and o2s directions). 0.0 disables boosting.
        """
        super().__init__()
        self.tau = float(temperature)
        self.hard_negatives = bool(hard_negatives)
        self.hard_negative_boost = float(hard_negative_boost)

    def forward(
        self,
        z_s: torch.Tensor,
        z_o: torch.Tensor,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute bidirectional InfoNCE.

        Args:
            z_s: Sonar embeddings ``(B, D)``, L2-normalized.
            z_o: Optical embeddings ``(B, D)``, L2-normalized.
            pos_weights: Optional ``(B,)`` weights for diagonal positives.

        Returns:
            ``(loss, diagnostics)`` where diagnostics includes similarity/weight stats.
        """
        b = z_s.size(0)
        device = z_s.device
        logits = torch.mm(z_s, z_o.T) / self.tau
        labels = torch.arange(b, device=device)
        if pos_weights is None:
            w = torch.ones(b, device=device, dtype=logits.dtype)
        else:
            w = pos_weights.clamp(min=0.4, max=1.0).to(device=device, dtype=logits.dtype)

        if self.hard_negatives and self.hard_negative_boost > 0 and b > 1:
            # Boost the hardest negative per anchor to increase pressure against confusing negatives.
            #
            # Important: logits may be fp16 under autocast. Using very large negative
            # sentinels (e.g. -1e9) overflows fp16. We therefore mine hard negatives
            # in fp32 with a safe finite mask value, then apply the boost to logits.
            eye = torch.eye(b, dtype=torch.bool, device=device)
            neg_mask_val = -1.0e4  # safe for fp16 and sufficient to remove diagonal from argmax
            logits_fp32 = logits.float()

            logits_s2o = logits_fp32.masked_fill(eye, neg_mask_val)
            hn_idx = logits_s2o.argmax(dim=1)
            logits[torch.arange(b, device=device), hn_idx] += self.hard_negative_boost

            logits_o2s = logits_fp32.T.masked_fill(eye, neg_mask_val)
            hn_idx_o = logits_o2s.argmax(dim=1)
            logits.T[torch.arange(b, device=device), hn_idx_o] += self.hard_negative_boost

        log_probs_s2o = F.log_softmax(logits, dim=1)
        log_probs_o2s = F.log_softmax(logits.T, dim=1)
        pos_s2o = log_probs_s2o[labels, labels]
        pos_o2s = log_probs_o2s[labels, labels]
        loss_s2o = -(w * pos_s2o).mean()
        loss_o2s = -(w * pos_o2s).mean()
        loss = (loss_s2o + loss_o2s) * 0.5

        with torch.no_grad():
            sim = torch.mm(z_s, z_o.T)
            diag = sim.diag()
            off_diag_mask = ~torch.eye(b, dtype=torch.bool, device=device)
            neg = sim[off_diag_mask]
            neg_mean = neg.mean().item() if neg.numel() > 0 else 0.0
            pos_mean = diag.mean().item() if diag.numel() > 0 else 0.0
            diagnostics = {
                "pos_sim": float(pos_mean),
                "neg_sim": float(neg_mean),
                "pos_minus_neg": float(pos_mean - neg_mean),
                "alignment": float((1.0 - diag).mean().item()) if diag.numel() > 0 else 0.0,
                "mean_weight": float(w.mean().item()),
            }
        return loss, diagnostics
