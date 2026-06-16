"""Composable matching pipeline for N07 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from geocam.matching.eval.metrics import control_point_errors, match_error_stats, mutual_information, ransac_stats
from geocam.matching.geometry import MatchSet, estimate_homography, nonrigid_refine_offset, shadow_filter, terrain_filter


@dataclass
class MatchingConfig:
    """Pipeline controls for staged ablations."""

    use_shadow_filter: bool = True
    use_terrain_filter: bool = True
    use_nonrigid_refine: bool = False
    top_k_candidates: int = 20
    adaptive_schedule: bool = True
    seed: int = 42


class MatchingPipeline:
    """Pipeline that can run baseline and enhanced variants."""

    def __init__(self, cfg: MatchingConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _mock_match(self, fixed_img: np.ndarray, moving_img: np.ndarray, n: int = 512) -> MatchSet:
        """Deterministic synthetic matcher placeholder.

        Replace this with SuperPoint/LightGlue integration in production runs.
        """
        h, w = fixed_img.shape[:2]
        fx = self.rng.uniform(0, w - 1, size=(n, 1))
        fy = self.rng.uniform(0, h - 1, size=(n, 1))
        base = np.concatenate([fx, fy], axis=1).astype(np.float32)
        noise = self.rng.normal(0.0, 8.0, size=base.shape).astype(np.float32)
        mov = base + noise
        score = np.clip(1.0 - (np.linalg.norm(noise, axis=1) / 40.0), 0.0, 1.0).astype(np.float32)
        return MatchSet(fixed_xy=base, moving_xy=mov, score=score)

    def run_case(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_shadow: Optional[np.ndarray] = None,
        moving_shadow: Optional[np.ndarray] = None,
        fixed_height: Optional[np.ndarray] = None,
        moving_height: Optional[np.ndarray] = None,
        control_points: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Run all enabled stages and return metrics."""
        matches = self._mock_match(fixed_img, moving_img)
        if self.cfg.use_shadow_filter and fixed_shadow is not None and moving_shadow is not None:
            matches = shadow_filter(matches, fixed_shadow, moving_shadow)
        if self.cfg.use_terrain_filter and fixed_height is not None and moving_height is not None:
            matches = terrain_filter(matches, fixed_height, moving_height)

        hmat, inlier_mask = estimate_homography(matches)
        refine = np.zeros(2, dtype=np.float32)
        if self.cfg.use_nonrigid_refine:
            refine = nonrigid_refine_offset(matches, inlier_mask)

        ref_xy = matches.moving_xy + refine.reshape(1, 2)
        out: Dict[str, float] = {}
        out.update(match_error_stats(matches.fixed_xy, matches.moving_xy, reference_xy=ref_xy))
        out.update(ransac_stats(inlier_mask, matches.score))
        if control_points is not None:
            out.update(control_point_errors(hmat, control_points))
        # Proxy registration MI: for now use input pair (replace with warped overlap in full integration).
        out["registration_mi"] = mutual_information(fixed_img, moving_img)
        return out


def run_matching_case(
    fixed_path: Path,
    moving_path: Path,
    pipeline: MatchingPipeline,
    control_points: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Load pair and run matching pipeline."""
    fixed = np.load(fixed_path).astype(np.float32)
    moving = np.load(moving_path).astype(np.float32)
    if fixed.ndim == 3:
        fixed = fixed.squeeze()
    if moving.ndim == 3:
        moving = moving.squeeze()
    zeros_f = np.zeros_like(fixed, dtype=np.uint8)
    zeros_m = np.zeros_like(moving, dtype=np.uint8)
    return pipeline.run_case(
        fixed_img=fixed,
        moving_img=moving,
        fixed_shadow=zeros_f,
        moving_shadow=zeros_m,
        fixed_height=fixed,
        moving_height=moving,
        control_points=control_points,
    )
