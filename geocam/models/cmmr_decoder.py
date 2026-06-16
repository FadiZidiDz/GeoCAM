"""Cross-modal masked reconstruction decoder (CMMR)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class CMMRDecoder(nn.Module):
    """Lightweight ViT decoder conditioned on optical embedding + visible sonar tokens."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialize decoder submodules from ``cfg['model']['cmmr']`` and sonar patch size.

        Args:
            cfg: Full GeoCAM config dict.
        """
        super().__init__()
        c = cfg["model"]["cmmr"]
        s = cfg["model"]["sonar"]
        self.patch_size = int(s["patch_size"])
        self.encoder_dim = int(s["embed_dim"])
        self.dec_dim = int(c["decoder_embed_dim"])
        depth = int(c["decoder_depth"])
        heads = int(c["decoder_num_heads"])
        p = cfg["model"]["projection_head"]
        self.z_opt_dim = int(p["out_dim"])
        self.patch_dim = self.patch_size * self.patch_size
        img_size = int(s["img_size"])
        num_patches = (img_size // self.patch_size) ** 2
        self.num_patches = num_patches

        self.z_opt_to_dec = nn.Linear(self.z_opt_dim, self.dec_dim)
        self.enc_to_dec = nn.Linear(self.encoder_dim, self.dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, self.dec_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        mlp_ratio = float(c.get("decoder_mlp_ratio", 4.0))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.dec_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.dec_norm = nn.LayerNorm(self.dec_dim)
        self.head = nn.Linear(self.dec_dim, self.patch_dim)

    @staticmethod
    def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Patchify a sonar tensor.

        Args:
            x: Tensor ``(B, C, H, W)`` with ``H,W`` divisible by ``patch_size``.
            patch_size: Patch side length.

        Returns:
            Tensor ``(B, L, patch_size**2 * C)``.
        """
        b, c, h, w = x.shape
        gh, gw = h // patch_size, w // patch_size
        x = x.reshape(b, c, gh, patch_size, gw, patch_size)
        x = torch.einsum("bchpwq->bhwcpq", x)
        return x.reshape(b, gh * gw, patch_size * patch_size * c)

    @staticmethod
    def unpatchify(patches: torch.Tensor, height: int, width: int, patch_size: int, channels: int) -> torch.Tensor:
        """Invert :meth:`patchify`.

        Args:
            patches: Tensor ``(B, L, patch_size**2 * C)``.
            height: Image height.
            width: Image width.
            patch_size: Patch side length.
            channels: Input channels (1 for sonar).

        Returns:
            Tensor ``(B, C, H, W)``.
        """
        b = patches.shape[0]
        gh, gw = height // patch_size, width // patch_size
        x = patches.reshape(b, gh, gw, patch_size, patch_size, channels)
        x = torch.einsum("bhwcpq->bchpwq", x)
        return x.reshape(b, channels, height, width)

    def forward(
        self,
        z_o: torch.Tensor,
        visible_sonar_tokens: torch.Tensor,
        mask: torch.Tensor,
        len_keep: int,
        ids_restore: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode masked sonar patches using optical conditioning.

        Args:
            z_o: Projected optical embedding ``(B, z_opt_dim)`` (contrastive space).
            visible_sonar_tokens: Encoder tokens for visible patches ``(B, len_keep, encoder_dim)``.
            mask: Boolean mask ``(B, L)`` with True at masked patches.
            len_keep: Number of visible patches (constant across batch).
            ids_restore: MAE restore indices ``(B, L)``.

        Returns:
            ``(pred_patches, masked_ids)`` where ``pred_patches`` is ``(B, n_mask, patch_dim)``
            aligned with ``masked_ids`` ``(B, n_mask)`` patch indices in original order positions.
        """
        b, _, _ = visible_sonar_tokens.shape
        l = mask.size(1)
        n_mask = l - len_keep

        ids_shuffle = torch.argsort(ids_restore, dim=1)
        masked_ids = ids_shuffle[:, len_keep:]

        z_tok = self.z_opt_to_dec(z_o).unsqueeze(1)
        vis_tok = self.enc_to_dec(visible_sonar_tokens)
        mask_tok = self.mask_token.expand(b, n_mask, -1)
        seq = torch.cat([z_tok, vis_tok, mask_tok], dim=1)
        if seq.size(1) != self.pos_embed.size(1):
            raise ValueError(
                f"Decoder sequence length {seq.size(1)} != pos_embed {self.pos_embed.size(1)}"
            )
        h = seq + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        h = self.dec_norm(h)
        mask_out = h[:, 1 + len_keep :, :]
        pred = self.head(mask_out)
        return pred, masked_ids
