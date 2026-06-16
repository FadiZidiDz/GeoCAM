"""Build markdown/csv paper artifacts from matching benchmark outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package matching artifacts")
    parser.add_argument("--runs_root", type=Path, default=Path("geocam_runs") / "matching_paper")
    args = parser.parse_args()

    summary_csv = args.runs_root / "benchmark_summary.csv"
    if not summary_csv.is_file():
        raise SystemExit(f"Missing {summary_csv}. Run run_matching_bench first.")
    rows = _read_csv(summary_csv)

    table_csv = args.runs_root / "matching_core_table.csv"
    table_md = args.runs_root / "matching_core_table.md"
    report_md = args.runs_root / "matching_report.md"

    cols = [
        "variant",
        "num_pairs",
        "match_mean_error_mean_std",
        "match_tol_ratio_mean_std",
        "inlier_ratio_mean_std",
        "mean_error_mean_std",
        "tolerance_ratio_mean_std",
        "registration_mi_mean_std",
    ]
    with table_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})

    with table_md.open("w", encoding="utf-8") as f:
        f.write("# Matching Core Table\n\n")
        f.write("| Variant | Pairs | MatchError | MatchTol | InlierRatio | RegError | RegTol | MI |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| "
                + f"{r.get('variant', '')} | {r.get('num_pairs', '')} | "
                + f"{r.get('match_mean_error_mean_std', '')} | "
                + f"{r.get('match_tol_ratio_mean_std', '')} | "
                + f"{r.get('inlier_ratio_mean_std', '')} | "
                + f"{r.get('mean_error_mean_std', '')} | "
                + f"{r.get('tolerance_ratio_mean_std', '')} | "
                + f"{r.get('registration_mi_mean_std', '')} |\n"
            )

    with report_md.open("w", encoding="utf-8") as f:
        f.write("# N07 Matching Benchmark Report\n\n")
        f.write("- This report compares configured variants on N07-style protocol.\n")
        f.write("- Metrics are mean±std over repeated protocol runs.\n\n")
        f.write(f"- Summary source: `{summary_csv}`\n")
        f.write(f"- Core table: `{table_md}` and `{table_csv}`\n")

    print(f"Wrote: {table_csv}")
    print(f"Wrote: {table_md}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
