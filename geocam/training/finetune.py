"""Full fine-tuning stub (decoder + head) for downstream segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn
import yaml

from geocam.utils.checkpoint import load_encoder_only
from geocam.utils.logger import set_seed


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Placeholder for UPerNet / SegFormer-style fine-tuning.

    After pre-training, unfreeze the sonar encoder with a small LR and attach a
    segmentation decoder; optimize on pixel-wise cross-entropy with ignore index 0.
    For labeled 384x384 tiles, apply explicit center-crop/resize to 256x256 before
    encoder forward. CMMR masking is pretraining-only and should be disabled here.
    """
    parser = argparse.ArgumentParser(description="Fine-tune GeoCAM encoder for segmentation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=False)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["data"]["seed"]))
    if args.checkpoint is None or not args.checkpoint.is_file():
        print("Provide --checkpoint from GeoCAM pre-training to initialize the sonar encoder.")
        return
    from geocam.models.geocam import GeoCAM

    model = GeoCAM(cfg)
    load_encoder_only(args.checkpoint, model)
    _ = nn.Identity()
    print("Encoder weights restored; plug in a segmentation decoder and labeled dataset.")


if __name__ == "__main__":
    main()
