"""Geometry-consistency filters and lightweight refinement utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MatchSet:
    """Sparse corresponding points."""

    fixed_xy: np.ndarray  # (N, 2)
    moving_xy: np.ndarray  # (N, 2)
    score: np.ndarray  # (N,)


def shadow_filter(matches: MatchSet, fixed_shadow: np.ndarray, moving_shadow: np.ndarray) -> MatchSet:
    """Drop matches that fall in shadow regions (1=shadow)."""
    fx = np.clip(matches.fixed_xy[:, 0].astype(np.int32), 0, fixed_shadow.shape[1] - 1)
    fy = np.clip(matches.fixed_xy[:, 1].astype(np.int32), 0, fixed_shadow.shape[0] - 1)
    mx = np.clip(matches.moving_xy[:, 0].astype(np.int32), 0, moving_shadow.shape[1] - 1)
    my = np.clip(matches.moving_xy[:, 1].astype(np.int32), 0, moving_shadow.shape[0] - 1)
    keep = (fixed_shadow[fy, fx] <= 0) & (moving_shadow[my, mx] <= 0)
    return MatchSet(
        fixed_xy=matches.fixed_xy[keep],
        moving_xy=matches.moving_xy[keep],
        score=matches.score[keep],
    )


def terrain_underestimation_mask(height: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Detect low-probability terrain regions via global z-score threshold."""
    h = height.astype(np.float32)
    mu = float(h.mean())
    std = float(h.std() + 1e-6)
    return (h < (mu - alpha * std)).astype(np.uint8)


def terrain_filter(matches: MatchSet, fixed_h: np.ndarray, moving_h: np.ndarray, alpha: float = 1.0) -> MatchSet:
    """Drop matches in terrain-underestimation masks."""
    fm = terrain_underestimation_mask(fixed_h, alpha=alpha)
    mm = terrain_underestimation_mask(moving_h, alpha=alpha)
    fx = np.clip(matches.fixed_xy[:, 0].astype(np.int32), 0, fm.shape[1] - 1)
    fy = np.clip(matches.fixed_xy[:, 1].astype(np.int32), 0, fm.shape[0] - 1)
    mx = np.clip(matches.moving_xy[:, 0].astype(np.int32), 0, mm.shape[1] - 1)
    my = np.clip(matches.moving_xy[:, 1].astype(np.int32), 0, mm.shape[0] - 1)
    keep = (fm[fy, fx] == 0) & (mm[my, mx] == 0)
    return MatchSet(
        fixed_xy=matches.fixed_xy[keep],
        moving_xy=matches.moving_xy[keep],
        score=matches.score[keep],
    )


def estimate_homography(matches: MatchSet) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate homography with OpenCV if available; fallback to identity."""
    if len(matches.score) < 4:
        return np.eye(3, dtype=np.float32), np.zeros(len(matches.score), dtype=np.uint8)
    try:
        import cv2  # type: ignore

        h, inlier = cv2.findHomography(matches.moving_xy, matches.fixed_xy, cv2.RANSAC, ransacReprojThreshold=8.0)
        if h is None:
            return np.eye(3, dtype=np.float32), np.zeros(len(matches.score), dtype=np.uint8)
        return h.astype(np.float32), inlier.reshape(-1).astype(np.uint8)
    except Exception:
        return np.eye(3, dtype=np.float32), np.zeros(len(matches.score), dtype=np.uint8)


def nonrigid_refine_offset(matches: MatchSet, inlier_mask: np.ndarray) -> np.ndarray:
    """Simple non-rigid proxy: confidence-weighted average residual offset.

    This is a lightweight refinement placeholder when TPS libraries are unavailable.
    """
    if matches.fixed_xy.size == 0 or inlier_mask.size == 0:
        return np.zeros(2, dtype=np.float32)
    keep = inlier_mask.astype(bool)
    if keep.sum() == 0:
        return np.zeros(2, dtype=np.float32)
    delta = matches.fixed_xy[keep] - matches.moving_xy[keep]
    w = matches.score[keep].reshape(-1, 1)
    w = w / (w.sum() + 1e-8)
    return (delta * w).sum(axis=0).astype(np.float32)
