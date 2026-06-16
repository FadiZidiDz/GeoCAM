"""Retrieval-guided pair gating and adaptive matching schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class GatingResult:
    """Top-K candidate list with scores."""

    query_id: str
    candidates: List[Tuple[str, float]]


def rank_candidates(
    query_id: str,
    query_embedding: np.ndarray,
    candidate_embeddings: Dict[str, np.ndarray],
) -> GatingResult:
    """Rank candidates by cosine similarity."""
    q = query_embedding.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    rows: List[Tuple[str, float]] = []
    for cid, emb in candidate_embeddings.items():
        v = emb.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        score = float(np.dot(q, v))
        rows.append((cid, score))
    rows.sort(key=lambda x: x[1], reverse=True)
    return GatingResult(query_id=query_id, candidates=rows)


def take_topk(gating: GatingResult, k: int) -> GatingResult:
    """Trim ranked candidates to top-K."""
    kk = max(1, int(k))
    return GatingResult(query_id=gating.query_id, candidates=gating.candidates[:kk])


def adaptive_schedule(similarity: float) -> Dict[str, float]:
    """Map retrieval confidence to matching hyperparameters.

    Higher retrieval confidence uses stricter keypoint filtering and fewer RANSAC iterations.
    """
    s = float(np.clip(similarity, -1.0, 1.0))
    s01 = 0.5 * (s + 1.0)
    keypoint_threshold = 0.0005 + (1.0 - s01) * 0.0015
    match_threshold = 0.10 + (1.0 - s01) * 0.10
    ransac_iters = int(2000 + (1.0 - s01) * 8000)
    return {
        "keypoint_threshold": float(keypoint_threshold),
        "match_threshold": float(match_threshold),
        "ransac_iters": float(ransac_iters),
    }


def filter_pairs_with_gating(
    protocol_pairs: Iterable[Tuple[str, str]],
    gated_candidates: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """Keep only pairs retained by gating."""
    out: List[Tuple[str, str]] = []
    for qid, cid in protocol_pairs:
        allow = gated_candidates.get(qid, [])
        if cid in allow:
            out.append((qid, cid))
    return out
