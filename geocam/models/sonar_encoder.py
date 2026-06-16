"""Single-channel ViT sonar encoder (timm) with optional MAE weight loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import timm


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    """Remove the first matching prefix from state dict keys."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if k.startswith(p):
                new_k = k[len(p) :]
                break
        out[new_k] = v
    return out


class SonarEncoder(nn.Module):
    """ViT-S/16 processing ``(B, 1, H, W)`` sonar tiles."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Build encoder from nested ``cfg['model']['sonar']``.

        Args:
            cfg: Full GeoCAM config dict.
        """
        super().__init__()
        s = cfg["model"]["sonar"]
        self.img_size = int(s["img_size"])
        self.patch_size = int(s["patch_size"])
        ckpt = s.get("mae_checkpoint") or ""
        # Fall back to ImageNet pre-trained weights when no MAE checkpoint is given.
        # Random init (pretrained=False) requires 200+ epochs to converge on small batches.
        use_imagenet = not bool(ckpt)
        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=use_imagenet,
            in_chans=1,
            num_classes=0,
            img_size=self.img_size,
        )
        if ckpt:
            path = Path(str(ckpt))
            if path.is_file():
                sd = torch.load(path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                if isinstance(sd, dict) and "model" in sd:
                    sd = sd["model"]
                sd = _strip_prefix(sd, ("module.", "model.", "encoder."))
                sd = _strip_prefix(sd, ("model.",))
                missing, unexpected = self.model.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    # Non-fatal: MAE checkpoints may omit head keys.
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS embedding ``(B, embed_dim)``.

        Args:
            x: Sonar tensor ``(B, 1, H, W)``.

        Returns:
            CLS token features.
        """
        feat = self.model.forward_features(x)
        if feat.ndim == 3:
            return feat[:, 0]
        return feat

    def forward_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens without CLS, shape ``(B, L, D)``.

        Args:
            x: Sonar tensor ``(B, 1, H, W)``.

        Returns:
            Patch token sequence from the last norm layer.
        """
        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        return x[:, 1:, :]

    def get_last_layer(self) -> nn.Module:
        """Return the final normalization module (for probe / fine-tune hooks).

        Returns:
            The ``norm`` layer of the timm ViT.
        """
        return self.model.norm
