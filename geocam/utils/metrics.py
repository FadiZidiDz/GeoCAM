"""Segmentation and retrieval metrics used by GeoCAM pipelines."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> Dict[str, float]:
    """Per-class IoU and mean IoU (excluding ``ignore_index``).

    Args:
        pred: Class indices ``(N,)`` or ``(H,W)``.
        target: Same shape as ``pred``.
        num_classes: Number of classes including ignore label.
        ignore_index: Label to exclude from mIoU.

    Returns:
        Dict with ``iou_class_{k}`` entries and ``miou``.
    """
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    ious: Dict[str, float] = {}
    accum = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else float("nan")
        ious[f"iou_class_{c}"] = iou
        if not np.isnan(iou):
            accum.append(iou)
    ious["miou"] = float(np.mean(accum)) if accum else 0.0
    return ious


class ConfusionMatrix:
    """Running confusion matrix for integer class predictions."""

    def __init__(self, num_classes: int) -> None:
        """Args:
            num_classes: Number of classes ``C``.
        """
        self.num_classes = int(num_classes)
        self.mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Add predictions from a batch.

        Args:
            pred: ``(N,)`` predicted class ids.
            target: ``(N,)`` ground-truth ids.
        """
        p = pred.detach().cpu().numpy().reshape(-1)
        t = target.detach().cpu().numpy().reshape(-1)
        for pi, ti in zip(p, t):
            self.mat[int(ti), int(pi)] += 1

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw matrix and per-class accuracy.

        Returns:
            ``(matrix, per_class_acc)``.
        """
        row_sum = self.mat.sum(axis=1).clip(min=1)
        per_class = np.diag(self.mat) / row_sum
        return self.mat, per_class


def _safe_topk(sim: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k indices with bounds checks."""
    if sim.numel() == 0:
        return torch.empty((sim.size(0), 0), dtype=torch.long, device=sim.device)
    kk = min(max(1, int(k)), sim.size(1))
    return sim.topk(kk, dim=1).indices


def recall_at_k(sim: torch.Tensor, target_idx: torch.Tensor, ks: Iterable[int]) -> Dict[str, float]:
    """Compute recall@k from a similarity matrix and target indices.

    Args:
        sim: Similarity matrix of shape ``(Q, C)``.
        target_idx: Correct candidate index for each query ``(Q,)``.
        ks: K values.

    Returns:
        Dict with keys ``R@k``.
    """
    if sim.size(0) == 0 or sim.size(1) == 0:
        return {f"R@{int(k)}": 0.0 for k in ks}
    out: Dict[str, float] = {}
    target = target_idx.view(-1, 1).to(sim.device)
    for k in ks:
        idx = _safe_topk(sim, int(k))
        hit = (idx == target).any(dim=1).float().mean().item()
        out[f"R@{int(k)}"] = float(hit)
    return out


def reciprocal_rank(sim: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """Per-query reciprocal rank for target candidate."""
    if sim.size(0) == 0 or sim.size(1) == 0:
        return torch.zeros(sim.size(0), dtype=torch.float32, device=sim.device)
    rank_order = torch.argsort(sim, dim=1, descending=True)
    target = target_idx.view(-1, 1).to(sim.device)
    match = rank_order == target
    # Exactly one target per row; argmax gives first (rank position).
    rank_pos = torch.argmax(match.to(torch.int64), dim=1) + 1
    return 1.0 / rank_pos.to(torch.float32)


def median_rank(sim: torch.Tensor, target_idx: torch.Tensor) -> float:
    """Median rank of the target index (1-indexed)."""
    if sim.size(0) == 0 or sim.size(1) == 0:
        return 0.0
    rank_order = torch.argsort(sim, dim=1, descending=True)
    target = target_idx.view(-1, 1).to(sim.device)
    match = rank_order == target
    rank_pos = torch.argmax(match.to(torch.int64), dim=1) + 1
    return float(torch.median(rank_pos.to(torch.float32)).item())


def ndcg_at_k(sim: torch.Tensor, relevance: torch.Tensor, k: int) -> float:
    """Compute mean NDCG@k.

    Args:
        sim: Similarity matrix ``(Q, C)``.
        relevance: Graded relevance matrix ``(Q, C)``.
        k: Truncation rank.
    """
    if sim.size(0) == 0 or sim.size(1) == 0:
        return 0.0
    kk = min(max(1, int(k)), sim.size(1))
    order = torch.argsort(sim, dim=1, descending=True)[:, :kk]
    rel_at_rank = torch.gather(relevance.to(sim.device), 1, order)
    discounts = 1.0 / torch.log2(torch.arange(2, kk + 2, device=sim.device, dtype=torch.float32))
    dcg = (rel_at_rank * discounts.unsqueeze(0)).sum(dim=1)
    ideal_order = torch.argsort(relevance.to(sim.device), dim=1, descending=True)[:, :kk]
    ideal_rel = torch.gather(relevance.to(sim.device), 1, ideal_order)
    idcg = (ideal_rel * discounts.unsqueeze(0)).sum(dim=1).clamp_min(1e-8)
    ndcg = dcg / idcg
    return float(ndcg.mean().item())


def retrieval_metrics_from_embeddings(
    z_q: torch.Tensor,
    z_c: torch.Tensor,
    ks: List[int],
    positive_weights: torch.Tensor | None = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute retrieval metrics for aligned query/candidate embeddings.

    Args:
        z_q: Query embeddings ``(N, D)``.
        z_c: Candidate embeddings ``(N, D)``.
        ks: K values for Recall/NDCG.
        positive_weights: Optional positive relevance per row ``(N,)``.
        prefix: Optional key prefix.
    """
    n = min(z_q.size(0), z_c.size(0))
    if n == 0:
        return {}
    sim = z_q[:n] @ z_c[:n].T
    target = torch.arange(n, device=sim.device, dtype=torch.long)
    metrics: Dict[str, float] = {}
    rec = recall_at_k(sim, target, ks)
    metrics.update(rec)
    rr = reciprocal_rank(sim, target)
    metrics["MRR"] = float(rr.mean().item())
    metrics["MedianRank"] = median_rank(sim, target)
    w = torch.ones(n, dtype=torch.float32, device=sim.device)
    if positive_weights is not None and positive_weights.numel() >= n:
        w = positive_weights[:n].to(sim.device).float().clamp_min(0.0)
    relevance = torch.zeros_like(sim, dtype=torch.float32)
    relevance[torch.arange(n, device=sim.device), target] = w
    for k in ks:
        metrics[f"NDCG@{int(k)}"] = ndcg_at_k(sim, relevance, int(k))
    if prefix:
        return {f"{prefix}{k}": float(v) for k, v in metrics.items()}
    return {k: float(v) for k, v in metrics.items()}
