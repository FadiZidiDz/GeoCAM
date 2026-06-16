"""Full GeoCAM module: sonar/optical encoders, SAA, contrastive projections, CMMR."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from geocam.models.cmmr_decoder import CMMRDecoder
from geocam.models.optical_encoder import OpticalEncoder
from geocam.models.projection_head import ProjectionHead
from geocam.models.saa import SetAttentionAggregator
from geocam.models.sonar_encoder import SonarEncoder


class GeoCAM(nn.Module):
    """Self-supervised cross-modal model for paired sonar tiles and optical sets."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Assemble encoders, aggregator, projection heads, and CMMR decoder.

        Args:
            cfg: Full GeoCAM config dict.
        """
        super().__init__()
        self.cfg = cfg
        self.sonar_encoder = SonarEncoder(cfg)
        self.optical_encoder = OpticalEncoder(cfg)
        self.saa = SetAttentionAggregator(cfg)
        self.use_saa = bool(cfg.get("model", {}).get("saa", {}).get("use_saa", True))
        self.use_geo_weights = bool(cfg.get("training", {}).get("use_geo_weights", True))
        in_dim = int(cfg["model"]["sonar"]["embed_dim"])
        self.sonar_proj = ProjectionHead(cfg, in_dim=in_dim)
        self.optical_proj = ProjectionHead(cfg, in_dim=in_dim)
        self.cmmr_decoder = CMMRDecoder(cfg)
        self.patch_size = int(cfg["model"]["sonar"]["patch_size"])
        self.mask_ratio = float(cfg["model"]["cmmr"]["mask_ratio"])

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """MAE-style patch masking in pixel space (zeroed patches).

        Args:
            x: Sonar tensor ``(B, 1, H, W)``.
            mask_ratio: Fraction of patches to mask.

        Returns:
            ``(x_masked, mask, ids_restore, len_keep)`` with ``mask`` True at masked patches.
        """
        b, _, h, w = x.shape
        p = self.patch_size
        l = (h // p) * (w // p)
        patches = CMMRDecoder.patchify(x, p)
        noise = torch.rand(b, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = int(l * (1.0 - mask_ratio))
        len_keep = max(1, min(len_keep, l - 1))
        mask = torch.ones(b, l, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :len_keep], False)
        patches_masked = patches.masked_fill(mask.unsqueeze(-1), 0.0)
        x_masked = CMMRDecoder.unpatchify(patches_masked, h, w, p, 1)
        return x_masked, mask, ids_restore, len_keep

    def gather_visible_tokens(
        self,
        patch_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
        len_keep: int,
    ) -> torch.Tensor:
        """Order visible patch tokens in MAE shuffle order.

        Args:
            patch_tokens: ``(B, L, D)`` encoder tokens (one per patch).
            ids_restore: ``(B, L)`` restore indices from :meth:`random_masking`.
            len_keep: Number of kept (visible) patches.

        Returns:
            Tensor ``(B, len_keep, D)``.
        """
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        visible_ids = ids_shuffle[:, :len_keep].unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
        return torch.gather(patch_tokens, 1, visible_ids)

    def encode_sonar(self, sonar: torch.Tensor) -> torch.Tensor:
        """Encode sonar tiles without CMMR masking (downstream use).

        Args:
            sonar: Sonar tensor ``(B, 1, H, W)``.

        Returns:
            Sonar encoder features ``(B, D)``.
        """
        return self.sonar_encoder(sonar)

    def forward(
        self,
        sonar: torch.Tensor,
        optical_set: torch.Tensor,
        weights: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        apply_cmmr_masking: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of tiles.

        Args:
            sonar: ``(B, 1, H, W)`` sonar tiles.
            optical_set: ``(B, N, 3, h, w)`` optical crops.
            weights: ``(B, N)`` overlap weights (0 for padded slots handled via mask).
            padding_mask: Bool ``(B, N)``, True where optical slots are padded.
            apply_cmmr_masking: If True, apply MAE-style masking for CMMR pretraining.

        Returns:
            Dict with contrastive embeddings, CMMR predictions, masks, and diagnostics.
        """
        b, n = optical_set.shape[:2]
        if apply_cmmr_masking:
            sonar_masked, mask, ids_restore, len_keep = self.random_masking(sonar, self.mask_ratio)
        else:
            sonar_masked = sonar
            l = (sonar.shape[-2] // self.patch_size) * (sonar.shape[-1] // self.patch_size)
            mask = torch.zeros(b, l, dtype=torch.bool, device=sonar.device)
            ids_restore = torch.arange(l, device=sonar.device).unsqueeze(0).repeat(b, 1)
            len_keep = l
        z_s_feat = self.sonar_encoder(sonar_masked)
        z_s = self.sonar_proj(z_s_feat)

        optical_flat = optical_set.reshape(b * n, *optical_set.shape[2:])
        o_feats = self.optical_encoder(optical_flat)
        o_feats = o_feats.view(b, n, -1)
        # Build per-image geo_weights for SAA attention bias.
        # Zero out padded slots so they don't influence the bias.
        valid = ~padding_mask if padding_mask is not None else torch.ones(b, n, dtype=torch.bool, device=sonar.device)
        geo_w = weights * valid.float() if self.use_geo_weights else valid.float()
        if self.use_saa:
            e_agg, attn_w = self.saa(o_feats, padding_mask, geo_weights=geo_w)
        else:
            # Mean-pool ablation baseline over valid optical views.
            denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            attn_simple = valid.float() / denom
            e_agg = (o_feats * attn_simple.unsqueeze(-1)).sum(dim=1)
            attn_w = attn_simple.unsqueeze(1)
        z_o = self.optical_proj(e_agg)

        if self.use_geo_weights:
            pos_weights = (weights * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp(min=1.0)
        else:
            pos_weights = torch.ones(b, device=sonar.device, dtype=weights.dtype)

        patch_tokens = self.sonar_encoder.forward_patch_tokens(sonar_masked)
        if apply_cmmr_masking:
            visible_tokens = self.gather_visible_tokens(patch_tokens, ids_restore, len_keep)
            pred_patches, masked_ids = self.cmmr_decoder(
                z_o,
                visible_tokens,
                mask,
                len_keep,
                ids_restore,
            )
        else:
            pred_patches = torch.empty(b, 0, self.patch_size * self.patch_size, device=sonar.device)
            masked_ids = torch.empty(b, 0, dtype=torch.long, device=sonar.device)

        return {
            "z_s": z_s,
            "z_o": z_o,
            "pos_weights": pos_weights,
            "pred_patches": pred_patches,
            "masked_ids": masked_ids,
            "mask": mask,
            "sonar_raw": sonar,
            "attn_weights": attn_w,
            "ids_restore": ids_restore,
            "len_keep": torch.tensor(len_keep, device=sonar.device),
        }
