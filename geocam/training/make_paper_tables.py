"""Build core ablation table (CSV + Markdown) for paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_variant_cfg(runs_root: Path, variant: str) -> Dict[str, Any]:
    variant_dir = runs_root / variant
    for seed_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
        cfg_path = seed_dir / "resolved_config.yaml"
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {}


def _bool_str(v: Any) -> str:
    return "yes" if bool(v) else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper ablation table from summaries")
    parser.add_argument("--runs_root", type=Path, default=Path("geocam_runs") / "paper_eval")
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    summary_csv = args.summary_csv or (args.runs_root / "paper_results_summary.csv")
    if not summary_csv.is_file():
        raise SystemExit(f"Missing summary CSV: {summary_csv}. Run aggregate_paper_results first.")
    rows = _load_summary(summary_csv)

    out_csv = args.out_csv or (args.runs_root / "core_ablation_table.csv")
    out_md = args.out_md or (args.runs_root / "core_ablation_table.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    table_rows: List[Dict[str, str]] = []
    for row in rows:
        variant = row["variant"]
        cfg = _load_variant_cfg(args.runs_root, variant)
        train = cfg.get("training", {})
        data = cfg.get("data", {})
        table_rows.append(
            {
                "method": variant,
                "CMMR": _bool_str(train.get("enable_cmmr_masking", False)),
                "overlap sampling": str(data.get("extension", {}).get("name", "none")),
                "temperature": str(train.get("temperature", "")),
                "hard-negatives": _bool_str(train.get("hard_negatives", False)),
                "R@1": row.get("val/R@1_mean_std", ""),
                "R@5": row.get("val/R@5_mean_std", ""),
                "R@10": row.get("val/R@10_mean_std", ""),
                "MRR": row.get("val/s2o/MRR_mean_std", ""),
                "MedianRank": row.get("val/s2o/MedianRank_mean_std", ""),
                "NDCG@5": row.get("val/s2o/NDCG@5_mean_std", ""),
                "pos_minus_neg": row.get("val/diag/pos_minus_neg_mean_std", ""),
            }
        )

    fields = [
        "method",
        "CMMR",
        "overlap sampling",
        "temperature",
        "hard-negatives",
        "R@1",
        "R@5",
        "R@10",
        "MRR",
        "MedianRank",
        "NDCG@5",
        "pos_minus_neg",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Core Ablation Table\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for r in table_rows:
            f.write("| " + " | ".join(r.get(k, "") for k in fields) + " |\n")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
