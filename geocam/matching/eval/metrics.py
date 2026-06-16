"""N07-style matching and registration metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np


def control_point_errors(
    homography: np.ndarray,
    control_points: np.ndarray,
    tol_px: float = 10.0,
) -> Dict[str, float]:
    """Evaluate control point projection error.

    control_points: (N,4) rows [fx, fy, mx, my].
    """
    if control_points.size == 0:
        return {"mean_error": 0.0, "std_error": 0.0, "tolerance_ratio": 0.0}
    fx = control_points[:, 0:2]
    mx = control_points[:, 2:4]
    ones = np.ones((mx.shape[0], 1), dtype=np.float32)
    mx_h = np.concatenate([mx, ones], axis=1)
    pred = (homography @ mx_h.T).T
    pred_xy = pred[:, :2] / np.clip(pred[:, 2:3], 1e-6, None)
    err = np.linalg.norm(pred_xy - fx, axis=1)
    return {
        "mean_error": float(err.mean()),
        "std_error": float(err.std()),
        "tolerance_ratio": float((err < tol_px).mean()),
    }


def match_error_stats(
    fixed_xy: np.ndarray,
    moving_xy: np.ndarray,
    reference_xy: np.ndarray | None = None,
    tol_px: float = 10.0,
) -> Dict[str, float]:
    """Compute matching error summary.

    If no reference field is provided, proxy error uses pairwise displacement magnitude.
    """
    if fixed_xy.size == 0:
        return {"match_mean_error": 0.0, "match_std_error": 0.0, "match_tol_ratio": 0.0}
    if reference_xy is None:
        err = np.linalg.norm(fixed_xy - moving_xy, axis=1)
    else:
        err = np.linalg.norm(fixed_xy - reference_xy, axis=1)
    return {
        "match_mean_error": float(err.mean()),
        "match_std_error": float(err.std()),
        "match_tol_ratio": float((err < tol_px).mean()),
    }


def ransac_stats(inlier_mask: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """Compute inlier count/ratio and confidence stats."""
    n = int(inlier_mask.size)
    if n == 0:
        return {
            "num_matches": 0.0,
            "num_inliers": 0.0,
            "inlier_ratio": 0.0,
            "mean_inlier_consistency": 0.0,
            "std_inlier_consistency": 0.0,
        }
    keep = inlier_mask.astype(bool)
    inlier_scores = score[keep] if keep.any() else np.array([0.0], dtype=np.float32)
    return {
        "num_matches": float(n),
        "num_inliers": float(int(keep.sum())),
        "inlier_ratio": float(keep.mean()),
        "mean_inlier_consistency": float(inlier_scores.mean()),
        "std_inlier_consistency": float(inlier_scores.std()),
    }


def mutual_information(fixed_img: np.ndarray, moving_warped: np.ndarray, bins: int = 64) -> float:
    """Compute MI in overlap domain."""
    x = fixed_img.astype(np.float32).reshape(-1)
    y = moving_warped.astype(np.float32).reshape(-1)
    hist2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist2d / np.clip(hist2d.sum(), 1e-8, None)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(np.clip(pxy[nz], 1e-12, None) / np.clip((px @ py)[nz], 1e-12, None)))
    return float(mi)
