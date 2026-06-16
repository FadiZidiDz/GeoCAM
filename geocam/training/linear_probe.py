"""Linear probe on frozen sonar encoder for labeled BenthiCat segmentation tiles.

This script converts pixel masks to tile labels via majority class over valid pixels
(``ignore_index`` excluded), then trains a linear classifier on top of frozen sonar
encoder features.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from geocam.utils.checkpoint import load_encoder_only
from geocam.utils.logger import set_seed


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resize_or_crop(
    x: torch.Tensor,
    size: int,
    mode: str,
    is_mask: bool = False,
) -> torch.Tensor:
    """Apply center-crop or resize to square size.

    Args:
        x: Tensor shape ``(C,H,W)``.
        size: Target side length.
        mode: Either ``center_crop`` or ``resize``.
        is_mask: Use nearest interpolation for masks.

    Returns:
        Tensor shape ``(C,size,size)``.
    """
    if mode == "center_crop":
        h, w = x.shape[-2:]
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        return x[..., top : top + size, left : left + size]
    if mode == "resize":
        interp_mode = "nearest" if is_mask else "bilinear"
        return F.interpolate(
            x.unsqueeze(0),
            size=(size, size),
            mode=interp_mode,
            align_corners=False if interp_mode == "bilinear" else None,
        ).squeeze(0)
    raise ValueError(f"Unsupported resize mode: {mode}")


class SonarSegTileDataset(Dataset):
    """Labeled tile dataset for linear probing.

    Expected layout:
    - ``sonar_dir/*.npy`` sonar tiles (384x384 or 256x256)
    - ``mask_dir/*.npy`` integer class masks with same stem as sonar tile
    """

    def __init__(
        self,
        sonar_dir: Path,
        mask_dir: Path,
        tile_size: int,
        resize_mode: str,
        ignore_index: int,
    ) -> None:
        self.sonar_dir = sonar_dir
        self.mask_dir = mask_dir
        self.tile_size = tile_size
        self.resize_mode = resize_mode
        self.ignore_index = ignore_index
        self.sonar_files = sorted(list(self.sonar_dir.glob("*.npy")))
        if not self.sonar_files:
            raise FileNotFoundError(f"No sonar .npy files in {self.sonar_dir}")

    def __len__(self) -> int:
        return len(self.sonar_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sonar_path = self.sonar_files[idx]
        mask_path = self.mask_dir / sonar_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {sonar_path.name}: {mask_path}")
        sonar = np.load(sonar_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)
        sonar_t = torch.from_numpy(sonar).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        sonar_t = _resize_or_crop(sonar_t, self.tile_size, self.resize_mode, is_mask=False)
        mask_t = _resize_or_crop(mask_t, self.tile_size, self.resize_mode, is_mask=True).squeeze(0).long()
        sonar_t = torch.clamp(sonar_t, 0.0, 1.0)
        valid = mask_t != self.ignore_index
        if valid.any():
            labels = mask_t[valid]
            tile_label = torch.mode(labels).values
        else:
            tile_label = torch.tensor(self.ignore_index, dtype=torch.long)
        return sonar_t, tile_label


@dataclass
class ProbeMetrics:
    """Container for linear probe metrics."""

    accuracy: float
    macro_f1: float


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
) -> ProbeMetrics:
    """Evaluate tile-level classifier metrics."""
    encoder.eval()
    head.eval()
    ys: List[int] = []
    ps: List[int] = []
    for sonar, label in loader:
        sonar = sonar.to(device)
        label = label.to(device)
        feats = encoder(sonar)
        logits = head(feats)
        pred = torch.argmax(logits, dim=1)
        valid = label != ignore_index
        ys.extend(label[valid].cpu().tolist())
        ps.extend(pred[valid].cpu().tolist())
    if not ys:
        return ProbeMetrics(accuracy=0.0, macro_f1=0.0)
    acc = float(accuracy_score(ys, ps))
    f1 = float(f1_score(ys, ps, average="macro"))
    return ProbeMetrics(accuracy=acc, macro_f1=f1)


def main() -> None:
    """Run linear probe on frozen sonar encoder.

    Notes:
    - 384x384 segmentation tiles are preprocessed by center-crop or resize to 256x256.
    - CMMR masking is a pretraining mechanism and is disabled here (encoder-only path).
    """
    parser = argparse.ArgumentParser(description="Linear probe (requires labels)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train_sonar_dir", type=Path, required=True)
    parser.add_argument("--train_mask_dir", type=Path, required=True)
    parser.add_argument("--val_sonar_dir", type=Path, required=True)
    parser.add_argument("--val_mask_dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["data"]["seed"]))
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    from geocam.models.geocam import GeoCAM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCAM(cfg).to(device)
    load_encoder_only(args.checkpoint, model)
    encoder = model.sonar_encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    ds_cfg = cfg["eval"].get("downstream", {})
    tile_size = int(ds_cfg.get("tile_size", 256))
    resize_mode = str(ds_cfg.get("resize_mode", "center_crop"))
    ignore_index = int(ds_cfg.get("ignore_index", 0))
    bs = int(ds_cfg.get("batch_size", 32))
    num_workers = int(ds_cfg.get("num_workers", 4))
    n_classes = int(cfg["eval"]["num_classes"])

    train_ds = SonarSegTileDataset(
        args.train_sonar_dir,
        args.train_mask_dir,
        tile_size=tile_size,
        resize_mode=resize_mode,
        ignore_index=ignore_index,
    )
    val_ds = SonarSegTileDataset(
        args.val_sonar_dir,
        args.val_mask_dir,
        tile_size=tile_size,
        resize_mode=resize_mode,
        ignore_index=ignore_index,
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    head = nn.Linear(int(cfg["model"]["sonar"]["embed_dim"]), n_classes).to(device)
    opt = torch.optim.SGD(
        head.parameters(),
        lr=float(cfg["eval"]["linear_probe_lr"]),
        momentum=float(cfg["eval"]["linear_probe_momentum"]),
    )
    epochs = int(cfg["eval"]["linear_probe_epochs"])
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    best_f1 = -1.0
    for epoch in range(epochs):
        head.train()
        running = 0.0
        count = 0
        for sonar, label in train_loader:
            sonar = sonar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.no_grad():
                feats = encoder(sonar)  # no CMMR masking in downstream eval/train
            logits = head(feats)
            loss = criterion(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            count += 1
        train_loss = running / max(1, count)
        metrics = evaluate(encoder, head, val_loader, device, ignore_index)
        print(
            f"epoch={epoch} train/loss={train_loss:.6f} "
            f"val/acc={metrics.accuracy:.6f} val/macro_f1={metrics.macro_f1:.6f}"
        )
        if metrics.macro_f1 > best_f1:
            best_f1 = metrics.macro_f1
            ckpt_dir = Path("geocam_runs") / "linear_probe"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"head": head.state_dict(), "best_macro_f1": best_f1}, ckpt_dir / "best_head.pt")
    print(f"best_macro_f1={best_f1:.6f}")


if __name__ == "__main__":
    main()
