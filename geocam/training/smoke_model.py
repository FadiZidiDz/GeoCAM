"""Fast GeoCAM sanity check: one batch forward + backward (no epoch loop).

Use your conda env with PyTorch/timm installed, e.g.::

    conda activate conda310
    cd /path/to/opti-acoustic-preprocessing-tools
    PYTHONPATH=. python -m geocam.training.smoke_model

Or::

    conda run -n conda310 python -m geocam.training.smoke_model
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from geocam.data.dataset import build_dataloaders
from geocam.losses.geo_infonce import GeoWeightedInfoNCE
from geocam.losses.reconstruction import PatchReconstructionLoss
from geocam.models.cmmr_decoder import CMMRDecoder
from geocam.models.geocam import GeoCAM


def _load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoCAM one-batch smoke test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["num_workers"] = 0

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"train_tiles={len(train_loader.dataset)} val_tiles={len(val_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = GeoCAM(cfg).to(device)
    infonce = GeoWeightedInfoNCE(temperature=float(cfg["training"]["temperature"])).to(device)
    recon = PatchReconstructionLoss().to(device)
    lam = float(cfg["training"]["lambda_cmmr"])

    batch = next(iter(train_loader))
    sonar, optical, weights, pad_mask, _ = batch
    sonar = sonar.to(device)
    optical = optical.to(device)
    weights = weights.to(device)
    pad_mask = pad_mask.to(device)

    model.train()
    out = model(sonar, optical, weights, pad_mask)
    target_patches = CMMRDecoder.patchify(sonar, model.patch_size)
    l_i, _ = infonce(out["z_s"], out["z_o"], out["pos_weights"])
    l_c = recon(out["pred_patches"], target_patches, out["masked_ids"])
    loss = l_i + lam * l_c
    loss.backward()

    g0 = next(model.sonar_encoder.parameters()).grad
    assert g0 is not None
    print(
        "OK",
        f"z_s={tuple(out['z_s'].shape)}",
        f"infonce={l_i.detach().item():.4f}",
        f"cmmr={l_c.detach().item():.4f}",
        f"total={loss.detach().item():.4f}",
    )


if __name__ == "__main__":
    main()
