"""Run true block-wise GeoCAM ablation on seed 42."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _deep_set(cfg: Dict[str, Any], key: str, value: Any) -> None:
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _run_one(cfg_path: Path, out_dir: Path, epochs: int, batch_size: int, num_workers: int) -> int:
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
        str(out_dir),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run block-wise ablation (seed 42)")
    parser.add_argument(
        "--base_config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "final_paper.yaml",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_root", type=Path, default=Path("geocam_runs") / "ablation_seed42_e10")
    args = parser.parse_args()

    base = _load_yaml(args.base_config)
    args.output_root.mkdir(parents=True, exist_ok=True)

    variants: List[Dict[str, Any]] = [
        {
            "name": "meanpool_contrastive_baseline",
            "changes": {
                "data.seed": 42,
                "model.saa.use_saa": False,
                "training.use_geo_weights": False,
                "training.enable_cmmr_masking": False,
                "training.lambda_cmmr": 0.0,
                "training.hard_negatives": False,
                "training.hard_negative_boost": 0.0,
            },
        },
        {
            "name": "saa_only",
            "changes": {
                "data.seed": 42,
                "model.saa.use_saa": True,
                "training.use_geo_weights": False,
                "training.enable_cmmr_masking": False,
                "training.lambda_cmmr": 0.0,
                "training.hard_negatives": False,
                "training.hard_negative_boost": 0.0,
            },
        },
        {
            "name": "saa_geo_weighted",
            "changes": {
                "data.seed": 42,
                "model.saa.use_saa": True,
                "training.use_geo_weights": True,
                "training.enable_cmmr_masking": False,
                "training.lambda_cmmr": 0.0,
                "training.hard_negatives": False,
                "training.hard_negative_boost": 0.0,
            },
        },
        {
            "name": "saa_geo_cmmr",
            "changes": {
                "data.seed": 42,
                "model.saa.use_saa": True,
                "training.use_geo_weights": True,
                "training.enable_cmmr_masking": True,
                "training.lambda_cmmr": 0.1,
                "training.hard_negatives": False,
                "training.hard_negative_boost": 0.0,
            },
        },
        {
            "name": "saa_geo_hardneg",
            "changes": {
                "data.seed": 42,
                "model.saa.use_saa": True,
                "training.use_geo_weights": True,
                "training.enable_cmmr_masking": False,
                "training.lambda_cmmr": 0.0,
                "training.hard_negatives": True,
                "training.hard_negative_boost": 0.7,
            },
        },
        {
            "name": "final_recipe",
            "changes": {
                "data.seed": 42,
                "model.saa.use_saa": True,
                "training.use_geo_weights": True,
                "training.enable_cmmr_masking": False,
                "training.lambda_cmmr": 0.0,
                "training.hard_negatives": False,
                "training.hard_negative_boost": 0.0,
            },
        },
    ]

    failures = 0
    with tempfile.TemporaryDirectory(prefix="geocam_block_ablation_") as tmp:
        tmp_dir = Path(tmp)
        for v in variants:
            cfg = yaml.safe_load(yaml.safe_dump(base))
            for k, val in v["changes"].items():
                _deep_set(cfg, k, val)
            cfg_path = tmp_dir / f"{v['name']}.yaml"
            _dump_yaml(cfg_path, cfg)
            out_dir = args.output_root / v["name"]
            rc = _run_one(cfg_path, out_dir, args.epochs, args.batch_size, args.num_workers)
            if rc != 0:
                failures += 1
                print(f"[FAILED] {v['name']} exit={rc}")
            else:
                print(f"[OK] {v['name']}")
    if failures:
        raise SystemExit(f"Finished with {failures} failed variant(s).")
    print("All ablation variants finished successfully.")


if __name__ == "__main__":
    main()
