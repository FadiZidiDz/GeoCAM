"""Run a small GeoCAM pretraining ablation matrix.

This utility writes temporary config variants and launches pretraining runs
with consistent seeds/splits for apples-to-apples comparison.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_cfg(path: Path, cfg: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_one(cfg_path: Path, output_dir: Path, epochs: int, batch_size: int, num_workers: int) -> int:
    cmd = [
        "python",
        "-m",
        "geocam.training.pretrain",
        "--config",
        str(cfg_path),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--output_dir",
        str(output_dir),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoCAM ablation matrix")
    parser.add_argument(
        "--base_config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_root", type=Path, default=Path("geocam_runs") / "ablations")
    args = parser.parse_args()

    base = _load_cfg(args.base_config)
    args.output_root.mkdir(parents=True, exist_ok=True)

    variants: List[Dict[str, Any]] = [
        {"name": "baseline", "changes": {}},
        {"name": "no_overlap_sampling", "changes": {"data.extension.name": "none"}},
        {"name": "no_cmmr", "changes": {"training.enable_cmmr_masking": False}},
        {
            "name": "no_overlap_no_cmmr",
            "changes": {"data.extension.name": "none", "training.enable_cmmr_masking": False},
        },
        {
            "name": "tau015_no_cmmr",
            "changes": {
                "training.temperature": 0.15,
                "training.enable_cmmr_masking": False,
                "data.extension.name": "none",
            },
        },
        {
            "name": "tau015_overlap_no_cmmr",
            "changes": {
                "training.temperature": 0.15,
                "training.enable_cmmr_masking": False,
                "data.extension.name": "overlap_sampling",
            },
        },
        {
            "name": "tau015_hardneg_no_cmmr",
            "changes": {
                "training.temperature": 0.15,
                "training.enable_cmmr_masking": False,
                "training.hard_negatives": True,
                "training.hard_negative_boost": 0.7,
                "data.extension.name": "none",
            },
        },
    ]

    with tempfile.TemporaryDirectory(prefix="geocam_ablation_") as tmp:
        tmp_dir = Path(tmp)
        failures = 0
        for variant in variants:
            cfg = yaml.safe_load(yaml.safe_dump(base))
            for key, value in variant["changes"].items():
                parts = key.split(".")
                cur = cfg
                for p in parts[:-1]:
                    cur = cur[p]
                cur[parts[-1]] = value
            cfg_path = tmp_dir / f"{variant['name']}.yaml"
            _dump_cfg(cfg_path, cfg)
            out_dir = args.output_root / variant["name"]
            rc = _run_one(cfg_path, out_dir, args.epochs, args.batch_size, args.num_workers)
            if rc != 0:
                failures += 1
                print(f"[FAILED] {variant['name']} (exit={rc})")
            else:
                print(f"[OK] {variant['name']}")
        if failures > 0:
            raise SystemExit(f"Ablations finished with {failures} failure(s).")
        print("All ablations completed successfully.")


if __name__ == "__main__":
    main()
