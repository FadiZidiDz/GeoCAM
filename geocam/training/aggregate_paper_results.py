"""Aggregate GeoCAM paper-eval runs into mean+-std summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _read_report(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    metrics = {}
    metrics.update(raw.get("best_metrics", {}))
    metrics.update({f"final::{k}": v for k, v in raw.get("final_metrics", {}).items()})
    metrics["best_epoch"] = float(raw.get("best_epoch", -1))
    metrics["best_val_loss"] = float(raw.get("best_val_loss", 0.0))
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _collect_runs(root: Path) -> Dict[str, List[Tuple[str, Dict[str, float]]]]:
    out: Dict[str, List[Tuple[str, Dict[str, float]]]] = {}
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows: List[Tuple[str, Dict[str, float]]] = []
        for seed_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            report = seed_dir / "training_report.json"
            if not report.is_file():
                continue
            rows.append((seed_dir.name, _read_report(report)))
        if rows:
            out[variant_dir.name] = rows
    return out


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _format_ms(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper-eval run metrics")
    parser.add_argument("--runs_root", type=Path, default=Path("geocam_runs") / "paper_eval")
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    runs = _collect_runs(args.runs_root)
    if not runs:
        raise SystemExit(f"No runs found under: {args.runs_root}")

    out_csv = args.out_csv or (args.runs_root / "paper_results_summary.csv")
    out_md = args.out_md or (args.runs_root / "paper_results_summary.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    keys_of_interest = [
        "val/R@1",
        "val/R@5",
        "val/R@10",
        "val/s2o/R@1",
        "val/s2o/R@5",
        "val/s2o/R@10",
        "val/o2s/R@1",
        "val/o2s/R@5",
        "val/o2s/R@10",
        "val/s2o/MRR",
        "val/o2s/MRR",
        "val/s2o/MedianRank",
        "val/o2s/MedianRank",
        "val/s2o/NDCG@5",
        "val/o2s/NDCG@5",
        "val/diag/pos_minus_neg",
        "val/diag/saa_entropy",
        "best_val_loss",
    ]

    rows_for_csv: List[Dict[str, str]] = []
    for variant, seed_rows in runs.items():
        metrics_by_key: Dict[str, List[float]] = {k: [] for k in keys_of_interest}
        seeds = []
        for seed_name, m in seed_rows:
            seeds.append(seed_name)
            for k in keys_of_interest:
                if k in m:
                    metrics_by_key[k].append(float(m[k]))
        row: Dict[str, str] = {
            "variant": variant,
            "num_seeds": str(len(seed_rows)),
            "seeds": ",".join(seeds),
        }
        for k in keys_of_interest:
            mean, std = _mean_std(metrics_by_key[k])
            row[f"{k}_mean"] = f"{mean:.6f}"
            row[f"{k}_std"] = f"{std:.6f}"
            row[f"{k}_mean_std"] = _format_ms(mean, std)
        rows_for_csv.append(row)

    fieldnames = sorted({k for r in rows_for_csv for k in r.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Paper Results Summary\n\n")
        f.write("| Variant | Seeds | R@1 | R@5 | R@10 | MRR (s2o/o2s) | MedianRank (s2o/o2s) | NDCG@5 (s2o/o2s) | pos_minus_neg |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_for_csv:
            f.write(
                "| "
                + f"{r['variant']} | {r['num_seeds']} | "
                + f"{r.get('val/R@1_mean_std', '0.0000 ± 0.0000')} | "
                + f"{r.get('val/R@5_mean_std', '0.0000 ± 0.0000')} | "
                + f"{r.get('val/R@10_mean_std', '0.0000 ± 0.0000')} | "
                + f"{r.get('val/s2o/MRR_mean_std', '0.0000 ± 0.0000')} / {r.get('val/o2s/MRR_mean_std', '0.0000 ± 0.0000')} | "
                + f"{r.get('val/s2o/MedianRank_mean_std', '0.0000 ± 0.0000')} / {r.get('val/o2s/MedianRank_mean_std', '0.0000 ± 0.0000')} | "
                + f"{r.get('val/s2o/NDCG@5_mean_std', '0.0000 ± 0.0000')} / {r.get('val/o2s/NDCG@5_mean_std', '0.0000 ± 0.0000')} | "
                + f"{r.get('val/diag/pos_minus_neg_mean_std', '0.0000 ± 0.0000')} |\n"
            )
        f.write("\n")
        f.write("All values are aggregated over available seed reports under each variant.\n")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
