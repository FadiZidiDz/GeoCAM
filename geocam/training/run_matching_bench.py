"""Run N07 matching benchmark variants with repeated protocol runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from geocam.matching.pipeline import MatchingConfig, MatchingPipeline, run_matching_case
from geocam.matching.protocol import load_control_points, load_pairs_csv, make_repeated_protocol


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matching benchmark variants")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "matching_bench.yaml",
    )
    parser.add_argument("--runs_root", type=Path, default=None)
    parser.add_argument("--limit_pairs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config).get("matching_bench", {})
    repo_root = Path(__file__).resolve().parents[2]
    protocol_csv = repo_root / cfg["protocol_csv"]
    pairs = load_pairs_csv(protocol_csv)
    repeats = int(cfg.get("repeats", 3))
    seed = int(cfg.get("seed", 42))
    protocol = make_repeated_protocol(pairs, repeats=repeats, seed=seed)
    if args.limit_pairs is not None:
        protocol = protocol[: max(1, int(args.limit_pairs))]

    cp_json = str(cfg.get("control_points_json", "")).strip()
    control_points = load_control_points(repo_root / cp_json) if cp_json else {}
    runs_root = args.runs_root or (repo_root / cfg.get("runs_root", "geocam_runs/matching_paper"))
    runs_root.mkdir(parents=True, exist_ok=True)

    variants = cfg.get("variants", [])
    if not variants:
        raise SystemExit("No variants configured in matching_bench.yaml")

    summary_rows: List[Dict[str, str]] = []
    for variant in variants:
        name = str(variant["name"])
        v_dir = runs_root / name
        v_dir.mkdir(parents=True, exist_ok=True)
        mcfg = MatchingConfig(
            use_shadow_filter=bool(variant.get("use_shadow_filter", True)),
            use_terrain_filter=bool(variant.get("use_terrain_filter", True)),
            use_nonrigid_refine=bool(variant.get("use_nonrigid_refine", False)),
            top_k_candidates=int(variant.get("top_k_candidates", 20)),
            adaptive_schedule=bool(variant.get("adaptive_schedule", True)),
            seed=seed,
        )
        pipe = MatchingPipeline(mcfg)
        case_rows: List[Dict[str, float | str]] = []
        metric_buckets: Dict[str, List[float]] = {}

        for p in protocol:
            row: Dict[str, float | str] = {
                "pair_id": p.pair_id,
                "fixed_path": p.fixed_path,
                "moving_path": p.moving_path,
            }
            if args.dry_run:
                metrics = {
                    "match_mean_error": 0.0,
                    "match_std_error": 0.0,
                    "match_tol_ratio": 0.0,
                    "num_matches": 0.0,
                    "num_inliers": 0.0,
                    "inlier_ratio": 0.0,
                    "mean_inlier_consistency": 0.0,
                    "std_inlier_consistency": 0.0,
                    "mean_error": 0.0,
                    "std_error": 0.0,
                    "tolerance_ratio": 0.0,
                    "registration_mi": 0.0,
                }
            else:
                cp = np.asarray(control_points.get(p.pair_id, []), dtype=np.float32) if control_points else None
                metrics = run_matching_case(
                    fixed_path=Path(p.fixed_path),
                    moving_path=Path(p.moving_path),
                    pipeline=pipe,
                    control_points=cp,
                )
            for k, v in metrics.items():
                row[k] = float(v)
                metric_buckets.setdefault(k, []).append(float(v))
            case_rows.append(row)

        case_csv = v_dir / "case_metrics.csv"
        with case_csv.open("w", encoding="utf-8", newline="") as f:
            fields = sorted({k for r in case_rows for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(case_rows)

        agg: Dict[str, Dict[str, float]] = {}
        for k, vals in metric_buckets.items():
            m, s = _mean_std(vals)
            agg[k] = {"mean": m, "std": s}

        with (v_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"variant": name, "num_pairs": len(case_rows), "metrics": agg}, f, indent=2)

        summary_row: Dict[str, str] = {"variant": name, "num_pairs": str(len(case_rows))}
        for k, ms in agg.items():
            summary_row[f"{k}_mean"] = f"{ms['mean']:.6f}"
            summary_row[f"{k}_std"] = f"{ms['std']:.6f}"
            summary_row[f"{k}_mean_std"] = f"{ms['mean']:.4f} ± {ms['std']:.4f}"
        summary_rows.append(summary_row)
        print(f"[OK] {name} -> {case_csv}")

    summary_csv = runs_root / "benchmark_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = sorted({k for r in summary_rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
