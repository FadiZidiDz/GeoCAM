"""Generate publication figures from GeoCAM paper-eval metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_epoch_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def _collect_variant_runs(runs_root: Path) -> Dict[str, List[List[Dict[str, float]]]]:
    out: Dict[str, List[List[Dict[str, float]]]] = {}
    for variant_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        runs: List[List[Dict[str, float]]] = []
        for seed_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            csv_path = seed_dir / "epoch_metrics.csv"
            if csv_path.is_file():
                rows = _load_epoch_csv(csv_path)
                if rows:
                    runs.append(rows)
        if runs:
            out[variant_dir.name] = runs
    return out


def _mean_curve(runs: List[List[Dict[str, float]]], key: str) -> tuple[np.ndarray, np.ndarray]:
    max_len = min(len(r) for r in runs)
    xs = np.arange(max_len, dtype=np.int64)
    vals = []
    for r in runs:
        vals.append([float(r[i].get(key, np.nan)) for i in range(max_len)])
    arr = np.asarray(vals, dtype=np.float64)
    return xs, np.nanmean(arr, axis=0)


def _plot_line_bundle(
    runs_by_variant: Dict[str, List[List[Dict[str, float]]]],
    keys: List[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for variant, runs in runs_by_variant.items():
            xs, mean = _mean_curve(runs, key)
            ax.plot(xs, mean, label=variant)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0][-1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pos_neg(runs_by_variant: Dict[str, List[List[Dict[str, float]]]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for variant, runs in runs_by_variant.items():
        _, pos = _mean_curve(runs, "val/diag/pos_sim")
        _, neg = _mean_curve(runs, "val/diag/neg_sim")
        ax.plot(neg, pos, marker="o", ms=2, label=variant)
    ax.set_xlabel("val/diag/neg_sim")
    ax.set_ylabel("val/diag/pos_sim")
    ax.set_title("Positive vs Negative Similarity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from epoch metrics")
    parser.add_argument("--runs_root", type=Path, default=Path("geocam_runs") / "paper_eval")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.runs_root / "figures")
    runs_by_variant = _collect_variant_runs(args.runs_root)
    if not runs_by_variant:
        raise SystemExit(f"No epoch_metrics.csv files found under {args.runs_root}")

    _plot_line_bundle(
        runs_by_variant,
        keys=["val/R@1", "val/R@5", "val/R@10"],
        title="Retrieval Curves",
        ylabel="Recall",
        out_path=out_dir / "retrieval_curves.png",
    )
    _plot_line_bundle(
        runs_by_variant,
        keys=["val/infonce"],
        title="Validation InfoNCE",
        ylabel="Loss",
        out_path=out_dir / "val_infonce_curve.png",
    )
    _plot_pos_neg(runs_by_variant, out_dir / "pos_vs_neg_similarity.png")
    _plot_line_bundle(
        runs_by_variant,
        keys=["val/diag/saa_entropy"],
        title="SAA Entropy",
        ylabel="Entropy",
        out_path=out_dir / "saa_entropy_curve.png",
    )
    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
