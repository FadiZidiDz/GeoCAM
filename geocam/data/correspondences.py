"""Load and normalize BenthiCat correspondences (path-based JSON → local tile stems)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def stem_from_any_path(path_str: str) -> str:
    """Return the filename stem (e.g. ``000042``) from a path string.

    Args:
        path_str: Absolute or relative path as stored in metadata.

    Returns:
        Filename stem without extension.
    """
    return Path(path_str).stem


def load_normalized_correspondences(
    correspondences_path: Path,
    data_root: Path,
    sonar_subdir: str,
    camera_subdir: str,
    verify_files: bool = True,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Load ``correspondences.json`` and map to local sonar/camera stems.

    Original release JSON uses absolute paths; this keeps **tile identity** as the
    sonar NPY stem (e.g. ``000000``) and optical identities as PNG stems.

    Args:
        correspondences_path: Path to ``correspondences.json``.
        data_root: Dataset root containing sonar and camera trees.
        sonar_subdir: Relative path segment to directory of ``*.npy`` tiles.
        camera_subdir: Relative path segment to directory of ``*.png`` images.
        verify_files: If True, skip tiles whose local ``{stem}.npy`` is missing.

    Returns:
        A pair ``(correspondences, tile_ids)`` where ``correspondences[tile_stem]``
        is a list of camera stems, and ``tile_ids`` is a sorted list of valid stems.

    Raises:
        FileNotFoundError: If the correspondences file does not exist.
    """
    if not correspondences_path.is_file():
        raise FileNotFoundError(correspondences_path)

    sonar_dir = data_root / sonar_subdir
    camera_dir = data_root / camera_subdir

    raw: Dict[str, Any]
    with correspondences_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, List[str]] = {}
    for sonar_key, img_list in raw.items():
        tile_stem = stem_from_any_path(str(sonar_key))
        if verify_files and not (sonar_dir / f"{tile_stem}.npy").is_file():
            continue
        cam_stems: List[str] = []
        for img_path in img_list:
            cstem = stem_from_any_path(str(img_path))
            if verify_files and not (camera_dir / f"{cstem}.png").is_file():
                continue
            cam_stems.append(cstem)
        if cam_stems:
            out[tile_stem] = cam_stems

    tile_ids = sorted(out.keys())
    return out, tile_ids
