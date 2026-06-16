"""Precompute geographic overlap weights for (sonar tile, optical image) pairs."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

from geocam.data.correspondences import load_normalized_correspondences, stem_from_any_path

SONAR_PATH_COL = "Patch_File_Path"
CAMERA_PATH_COL = "Image_File_Path"
CORNER_COLS = [
    "Top_Left_East",
    "Top_Left_North",
    "Top_Right_East",
    "Top_Right_North",
    "Bottom_Right_East",
    "Bottom_Right_North",
    "Bottom_Left_East",
    "Bottom_Left_North",
]


def _row_polygon(row: pd.Series) -> Polygon:
    """Build a Shapely polygon from a CSV row with eight UTM corner columns."""
    coords = [
        (float(row[CORNER_COLS[0]]), float(row[CORNER_COLS[1]])),
        (float(row[CORNER_COLS[2]]), float(row[CORNER_COLS[3]])),
        (float(row[CORNER_COLS[4]]), float(row[CORNER_COLS[5]])),
        (float(row[CORNER_COLS[6]]), float(row[CORNER_COLS[7]])),
    ]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly


def _load_footprint_maps(
    sonar_csv: Path,
    camera_csv: Path,
) -> Tuple[Dict[str, Polygon], Dict[str, Polygon]]:
    """Load stem → footprint polygon maps from BenthiCat-style CSVs."""
    s_df = pd.read_csv(sonar_csv)
    c_df = pd.read_csv(camera_csv)
    sonar_map: Dict[str, Polygon] = {}
    cam_map: Dict[str, Polygon] = {}
    for _, row in s_df.iterrows():
        stem = stem_from_any_path(str(row[SONAR_PATH_COL]))
        sonar_map[stem] = _row_polygon(row)
    for _, row in c_df.iterrows():
        stem = stem_from_any_path(str(row[CAMERA_PATH_COL]))
        cam_map[stem] = _row_polygon(row)
    return sonar_map, cam_map


_SONAR_MAP: Dict[str, Polygon] = {}
_CAM_MAP: Dict[str, Polygon] = {}


def _pool_init(sonar_csv: str, camera_csv: str) -> None:
    """Load footprint maps once per worker process (avoids huge pickling overhead)."""
    global _SONAR_MAP, _CAM_MAP
    sm, cm = _load_footprint_maps(Path(sonar_csv), Path(camera_csv))
    _SONAR_MAP = sm
    _CAM_MAP = cm


def _pair_weight(
    pair: Tuple[str, str],
    sonar_map: Dict[str, Polygon],
    cam_map: Dict[str, Polygon],
) -> Optional[Tuple[str, str, float]]:
    """Compute overlap ratio for one (tile_stem, cam_stem) pair."""
    tile_stem, cam_stem = pair
    if tile_stem not in sonar_map or cam_stem not in cam_map:
        return None
    poly_s = sonar_map[tile_stem]
    poly_c = cam_map[cam_stem]
    if poly_c.area <= 0:
        return tile_stem, cam_stem, 0.0
    try:
        inter = poly_s.intersection(poly_c).area
    except Exception:
        inter = 0.0
    ratio = float(inter / poly_c.area)
    ratio = max(0.0, min(1.0, ratio))
    return tile_stem, cam_stem, ratio


def _pair_weight_worker(pair: Tuple[str, str]) -> Optional[Tuple[str, str, float]]:
    """Multiprocessing entry using process-local footprint maps."""
    return _pair_weight(pair, _SONAR_MAP, _CAM_MAP)


def build_pair_list(correspondences: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Flatten correspondences to (tile_stem, cam_stem) pairs."""
    pairs: List[Tuple[str, str]] = []
    for t, cams in correspondences.items():
        for c in cams:
            pairs.append((t, c))
    return pairs


def compute_overlap_weights(
    correspondences: Dict[str, List[str]],
    sonar_map: Dict[str, Polygon],
    cam_map: Dict[str, Polygon],
    num_workers: int = 4,
    sonar_csv: Optional[Path] = None,
    camera_csv: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute overlap ratio intersection/area(optical) for every linked pair.

    Args:
        correspondences: tile stem → list of camera stems.
        sonar_map: tile stem → sonar footprint polygon.
        cam_map: camera stem → optical footprint polygon.
        num_workers: worker processes for pair evaluation.
        sonar_csv: Path passed to worker initializer when ``num_workers > 1``.
        camera_csv: Path passed to worker initializer when ``num_workers > 1``.

    Returns:
        Nested dict ``overlap[tile_stem][cam_stem] = weight``.
    """
    pairs = build_pair_list(correspondences)
    overlap: Dict[str, Dict[str, float]] = {}

    if num_workers <= 1:
        for t, c in tqdm(pairs, desc="overlap"):
            r = _pair_weight((t, c), sonar_map, cam_map)
            if r is None:
                continue
            ts, cs, w = r
            overlap.setdefault(ts, {})[cs] = w
    else:
        if sonar_csv is None or camera_csv is None:
            raise ValueError("sonar_csv and camera_csv are required when num_workers > 1")
        with Pool(
            processes=num_workers,
            initializer=_pool_init,
            initargs=(str(sonar_csv), str(camera_csv)),
        ) as pool:
            for r in tqdm(
                pool.imap_unordered(_pair_weight_worker, pairs, chunksize=512),
                total=len(pairs),
                desc="overlap",
            ):
                if r is None:
                    continue
                ts, cs, w = r
                overlap.setdefault(ts, {})[cs] = w
    return overlap


def main() -> None:
    """CLI entry: read CSVs + correspondences, write ``overlap_weights.json``."""
    parser = argparse.ArgumentParser(description="Precompute overlap weights for GeoCAM.")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Dataset root (parent of sonar/camera subtrees).",
    )
    parser.add_argument(
        "--correspondences",
        type=Path,
        default=None,
        help="Path to correspondences.json (default: data_root/correspondences.json).",
    )
    parser.add_argument(
        "--sonar_csv",
        type=Path,
        default=None,
        help="Path to sonar.csv (default: data_root/sonar/sonar/sonar.csv).",
    )
    parser.add_argument(
        "--camera_csv",
        type=Path,
        default=None,
        help="Path to camera.csv (default: data_root/camera/camera/camera.csv).",
    )
    parser.add_argument(
        "--sonar_subdir",
        type=str,
        default="sonar/sonar",
        help="Relative path to sonar npy directory under data_root.",
    )
    parser.add_argument(
        "--camera_subdir",
        type=str,
        default="camera/camera",
        help="Relative path to camera png directory under data_root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data_root/overlap_weights.json).",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    data_root: Path = args.data_root
    corr_path = args.correspondences or (data_root / "correspondences.json")
    sonar_csv = args.sonar_csv or (data_root / "sonar" / "sonar" / "sonar.csv")
    if args.sonar_csv is None and not sonar_csv.is_file():
        legacy_sonar_csv = data_root / "sona" / "sonar" / "sonar.csv"
        if legacy_sonar_csv.is_file():
            sonar_csv = legacy_sonar_csv
    camera_csv = args.camera_csv or (data_root / "camera" / "camera" / "camera.csv")
    out_path = args.output or (data_root / "overlap_weights.json")

    corr, _ = load_normalized_correspondences(
        corr_path,
        data_root,
        args.sonar_subdir,
        args.camera_subdir,
        verify_files=True,
    )
    sonar_map, cam_map = _load_footprint_maps(sonar_csv, camera_csv)
    weights = compute_overlap_weights(
        corr,
        sonar_map,
        cam_map,
        num_workers=args.num_workers,
        sonar_csv=sonar_csv,
        camera_csv=camera_csv,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(weights, f)
    print(f"Wrote {out_path} ({len(weights)} tiles with weights).")


if __name__ == "__main__":
    main()
