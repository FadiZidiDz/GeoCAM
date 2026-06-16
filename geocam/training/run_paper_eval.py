"""Run multi-seed full-budget GeoCAM paper evaluation."""

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


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _deep_set(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _run_pretrain(
    cfg_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> int:
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
    if max_train_batches is not None:
        cmd += ["--max_train_batches", str(max_train_batches)]
    if max_val_batches is not None:
        cmd += ["--max_val_batches", str(max_val_batches)]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _resolve_variant_cfg(repo_root: Path, cfg_rel: str) -> Path:
    p = Path(cfg_rel)
    if p.is_absolute():
        return p
    return repo_root / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoCAM full paper protocol")
    parser.add_argument(
        "--paper_config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "paper_eval.yaml",
    )
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    conf = _load_yaml(args.paper_config).get("paper_eval", {})
    seeds: List[int] = [int(s) for s in conf.get("seeds", [42, 52, 62])]
    variants: List[Dict[str, Any]] = list(conf.get("variants", []))
    include_hardneg = bool(conf.get("include_hardneg_tuned", False))
    if include_hardneg:
        variants.append(
            {
                "name": "hardneg_tuned",
                "config": "geocam/configs/final_paper.yaml",
                "overrides": {
                    "training.hard_negatives": True,
                    "training.hard_negative_boost": 0.7,
                },
            }
        )
    if not variants:
        raise ValueError("paper_eval.variants is empty.")

    output_root = args.output_root or (repo_root / conf.get("output_root", "geocam_runs/paper_eval"))
    output_root.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or conf.get("default_epochs", 100))
    batch_size = int(args.batch_size or conf.get("default_batch_size", 64))
    num_workers = int(args.num_workers or conf.get("default_num_workers", 8))

    failures: List[str] = []
    with tempfile.TemporaryDirectory(prefix="geocam_paper_eval_") as tmp:
        tmp_dir = Path(tmp)
        for variant in variants:
            name = str(variant["name"])
            src_cfg = _resolve_variant_cfg(repo_root, str(variant["config"]))
            base_cfg = _load_yaml(src_cfg)
            overrides: Dict[str, Any] = variant.get("overrides", {}) or {}
            for seed in seeds:
                run_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
                _deep_set(run_cfg, "data.seed", int(seed))
                for key, value in overrides.items():
                    _deep_set(run_cfg, key, value)
                cfg_path = tmp_dir / f"{name}_seed{seed}.yaml"
                _dump_yaml(cfg_path, run_cfg)
                out_dir = output_root / name / f"seed_{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)
                _dump_yaml(out_dir / "resolved_config.yaml", run_cfg)
                if args.dry_run:
                    print(f"[DRY] variant={name} seed={seed} cfg={cfg_path} out={out_dir}")
                    continue
                rc = _run_pretrain(
                    cfg_path,
                    out_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_train_batches=args.max_train_batches,
                    max_val_batches=args.max_val_batches,
                )
                if rc != 0:
                    failures.append(f"{name}/seed_{seed} (exit={rc})")
                    print(f"[FAILED] {name} seed={seed}")
                else:
                    print(f"[OK] {name} seed={seed}")
    if failures:
        joined = "\n".join(failures)
        raise SystemExit(f"Paper evaluation completed with failures:\n{joined}")
    print("Paper evaluation completed successfully.")


if __name__ == "__main__":
    main()
