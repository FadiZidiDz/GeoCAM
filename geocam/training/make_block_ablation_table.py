"""Build markdown/csv/LaTeX tables from seed-42 block ablation runs.

Row order matches ``run_block_ablation_seed42.py`` through the hard-negative variant.
The last row is **GeoCAM (full model)** with all component columns marked; retrieval
numbers are aggregated from ``paper_eval/final_paper/seed_*/`` (full training budget),
not from the short ablation folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Folder under --runs_root, or None -> metrics from --paper_final_root (multi-seed).
# (folder, label, SAA, GeoW InfoNCE, CMMR, HardNeg, temperature)
VARIANT_ROWS: List[Tuple[Optional[str], str, bool, bool, bool, bool, str]] = [
    ("meanpool_contrastive_baseline", "MeanPool Contrastive Baseline", False, False, False, False, "0.15"),
    ("saa_only", "SAA Only", True, False, False, False, "0.15"),
    ("saa_geo_weighted", "SAA + Geo-Weighted InfoNCE", True, True, False, False, "0.15"),
    ("saa_geo_cmmr", "SAA + Geo-Weighted InfoNCE + CMMR", True, True, True, False, "0.15"),
    ("saa_geo_hardneg", "SAA + Geo-Weighted InfoNCE + HardNeg", True, True, False, True, "0.15"),
    (None, "GeoCAM (full model)", True, True, True, True, "0.15"),
]


def _read_report(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    bm = raw.get("best_metrics", {})
    return {
        "R@1": float(bm.get("val/R@1", 0.0)),
        "R@5": float(bm.get("val/R@5", 0.0)),
        "R@10": float(bm.get("val/R@10", 0.0)),
        "MRR": float(bm.get("val/s2o/MRR", 0.0)),
        "NDCG@5": float(bm.get("val/s2o/NDCG@5", 0.0)),
        "pos_minus_neg": float(bm.get("val/diag/pos_minus_neg", 0.0)),
    }


def _tick(on: bool) -> str:
    return r"\checkmark" if on else "X"


def _aggregate_paper_final(paper_root: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (means, stds) for keys R@1, R@5, R@10, MRR, NDCG@5, pos_minus_neg."""
    reports = sorted(paper_root.glob("seed_*/training_report.json"))
    if not reports:
        raise SystemExit(f"No seed_*/training_report.json under {paper_root}")
    keys = ["R@1", "R@5", "R@10", "MRR", "NDCG@5", "pos_minus_neg"]
    series: Dict[str, List[float]] = {k: [] for k in keys}
    for path in reports:
        m = _read_report(path)
        for k in keys:
            series[k].append(m[k])
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for k, vals in series.items():
        arr = np.asarray(vals, dtype=np.float64)
        means[k] = float(arr.mean())
        stds[k] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return means, stds


def _latex_table(rows: List[Dict[str, str]], caption_suffix: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Block-wise ablation of GeoCAM components ({caption_suffix}). \checkmark: enabled, X: disabled.}}",
        r"\label{tab:block_ablation_named}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lccccccccccc}",
        r"\hline",
        r"Method Variant & SAA & GeoW InfoNCE & CMMR & HardNeg & Temp & R@1 & R@5 & R@10 & MRR & NDCG@5 & pos\_minus\_neg \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['saa']} & {r['geo']} & {r['cmmr']} & {r['hard']} & {r['temp']} "
            f"& {r['R@1']} & {r['R@5']} & {r['R@10']} & {r['MRR']} & {r['NDCG@5']} & {r['pos_minus_neg']} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create block ablation table from reports")
    parser.add_argument("--runs_root", type=Path, default=Path("geocam_runs") / "ablation_seed42_e10")
    parser.add_argument(
        "--paper_final_root",
        type=Path,
        default=Path("geocam_runs") / "paper_eval" / "final_paper",
        help="Directory with seed_*/training_report.json for the full GeoCAM row",
    )
    parser.add_argument(
        "--caption_suffix",
        type=str,
        default=(
            r"ablations: seed 42, 10 epochs, val @ best ckpt; "
            r"GeoCAM: mean$\pm$std over \texttt{paper\_eval/final\_paper} seeds, val @ best ckpt"
        ),
        help="Inserted into the LaTeX \\caption{...}",
    )
    args = parser.parse_args()

    if not args.runs_root.is_dir():
        raise SystemExit(f"Runs root not found: {args.runs_root}")

    paper_means: Optional[Dict[str, float]] = None
    paper_stds: Optional[Dict[str, float]] = None
    if any(folder is None for folder, *_ in VARIANT_ROWS):
        if not args.paper_final_root.is_dir():
            raise SystemExit(f"Paper final root not found: {args.paper_final_root}")
        paper_means, paper_stds = _aggregate_paper_final(args.paper_final_root)

    rows_simple: List[Dict[str, str]] = []
    rows_latex: List[Dict[str, str]] = []

    for folder, label, saa, geo, cmmr, hard, temp in VARIANT_ROWS:
        if folder is not None:
            report = args.runs_root / folder / "training_report.json"
            if not report.is_file():
                raise SystemExit(f"Missing report: {report}")
            m = _read_report(report)
            r1 = f"{m['R@1']:.4f}"
            r5 = f"{m['R@5']:.4f}"
            r10 = f"{m['R@10']:.4f}"
            mrr = f"{m['MRR']:.4f}"
            ndcg = f"{m['NDCG@5']:.4f}"
            pmn = f"{m['pos_minus_neg']:.4f}"
            variant_key = folder
            r1_tex, r5_tex, r10_tex = r1, r5, r10
            mrr_tex, ndcg_tex, pmn_tex = mrr, ndcg, pmn
        else:
            assert paper_means is not None and paper_stds is not None
            variant_key = "paper_eval_final_paper_mean_std"
            r1 = f"{paper_means['R@1']:.4f} ± {paper_stds['R@1']:.4f}"
            r5 = f"{paper_means['R@5']:.4f} ± {paper_stds['R@5']:.4f}"
            r10 = f"{paper_means['R@10']:.4f} ± {paper_stds['R@10']:.4f}"
            mrr = f"{paper_means['MRR']:.4f} ± {paper_stds['MRR']:.4f}"
            ndcg = f"{paper_means['NDCG@5']:.4f} ± {paper_stds['NDCG@5']:.4f}"
            pmn = f"{paper_means['pos_minus_neg']:.4f} ± {paper_stds['pos_minus_neg']:.4f}"
            r1_tex = f"${paper_means['R@1']:.4f} \\pm {paper_stds['R@1']:.4f}$"
            r5_tex = f"${paper_means['R@5']:.4f} \\pm {paper_stds['R@5']:.4f}$"
            r10_tex = f"${paper_means['R@10']:.4f} \\pm {paper_stds['R@10']:.4f}$"
            mrr_tex = f"${paper_means['MRR']:.4f} \\pm {paper_stds['MRR']:.4f}$"
            ndcg_tex = f"${paper_means['NDCG@5']:.4f} \\pm {paper_stds['NDCG@5']:.4f}$"
            pmn_tex = f"${paper_means['pos_minus_neg']:.4f} \\pm {paper_stds['pos_minus_neg']:.4f}$"

        rows_simple.append(
            {
                "variant": variant_key,
                "R@1": r1,
                "R@5": r5,
                "R@10": r10,
                "MRR": mrr,
                "NDCG@5": ndcg,
                "pos_minus_neg": pmn,
            }
        )
        rows_latex.append(
            {
                "label": label,
                "saa": _tick(saa),
                "geo": _tick(geo),
                "cmmr": _tick(cmmr),
                "hard": _tick(hard),
                "temp": temp,
                "R@1": r1_tex,
                "R@5": r5_tex,
                "R@10": r10_tex,
                "MRR": mrr_tex,
                "NDCG@5": ndcg_tex,
                "pos_minus_neg": pmn_tex,
            }
        )

    out_csv = args.runs_root / "block_ablation_seed42_e10.csv"
    out_md = args.runs_root / "block_ablation_seed42_e10.md"
    out_tex = args.runs_root / "block_ablation_seed42_e10.tex"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["variant", "R@1", "R@5", "R@10", "MRR", "NDCG@5", "pos_minus_neg"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_simple)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Block ablation (10-epoch seed-42 runs) + GeoCAM paper-eval (multi-seed)\n\n")
        f.write("| Variant | R@1 | R@5 | R@10 | MRR | NDCG@5 | pos\\_minus\\_neg |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_simple:
            f.write(
                f"| {r['variant']} | {r['R@1']} | {r['R@5']} | {r['R@10']} | {r['MRR']} | {r['NDCG@5']} | {r['pos_minus_neg']} |\n"
            )

    out_tex.write_text(_latex_table(rows_latex, args.caption_suffix), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_tex}")


if __name__ == "__main__":
    main()
