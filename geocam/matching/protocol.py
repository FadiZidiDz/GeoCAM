"""Protocol helpers for N07-style pair construction and metadata loading."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class MatchPair:
    """One fixed/moving pair used for matching benchmark."""

    pair_id: str
    fixed_path: str
    moving_path: str
    sector: str = "N07"


def load_pairs_csv(path: Path) -> List[MatchPair]:
    """Load pair protocol from CSV.

    Required columns: pair_id, fixed_path, moving_path.
    Optional: sector.
    """
    pairs: List[MatchPair] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                MatchPair(
                    pair_id=str(row["pair_id"]),
                    fixed_path=str(row["fixed_path"]),
                    moving_path=str(row["moving_path"]),
                    sector=str(row.get("sector", "N07")),
                )
            )
    return pairs


def save_pairs_csv(path: Path, pairs: List[MatchPair]) -> None:
    """Write pair protocol to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "fixed_path", "moving_path", "sector"])
        writer.writeheader()
        for p in pairs:
            writer.writerow(
                {
                    "pair_id": p.pair_id,
                    "fixed_path": p.fixed_path,
                    "moving_path": p.moving_path,
                    "sector": p.sector,
                }
            )


def make_repeated_protocol(pairs: List[MatchPair], repeats: int, seed: int) -> List[MatchPair]:
    """Repeat and shuffle a pair list for robust repeated runs."""
    if repeats <= 1:
        return list(pairs)
    rng = np.random.default_rng(seed)
    out: List[MatchPair] = []
    for r in range(repeats):
        idx = np.arange(len(pairs))
        rng.shuffle(idx)
        for i in idx:
            p = pairs[int(i)]
            out.append(
                MatchPair(
                    pair_id=f"{p.pair_id}_r{r}",
                    fixed_path=p.fixed_path,
                    moving_path=p.moving_path,
                    sector=p.sector,
                )
            )
    return out


def load_control_points(path: Path) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """Load manual control points JSON.

    Format:
    {
      "pair_id": [[fx, fy, mx, my], ...]
    }
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for k, vals in raw.items():
        out[str(k)] = [tuple(float(x) for x in row) for row in vals]
    return out
