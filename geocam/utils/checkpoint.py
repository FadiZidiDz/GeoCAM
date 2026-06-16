"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training state to disk.

    Args:
        model: Model to serialize.
        optimizer: Optimizer state.
        epoch: Current epoch index.
        loss: Scalar loss (e.g. validation).
        path: Output ``.pt`` path.
        extra: Optional extra fields merged into the checkpoint dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "loss": float(loss),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float]:
    """Load weights and optimizer state.

    Args:
        path: Checkpoint path.
        model: Model whose ``load_state_dict`` is called (non-strict).
        optimizer: Optional optimizer to restore.

    Returns:
        ``(epoch, loss)`` from checkpoint metadata.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("loss", 0.0))


def load_encoder_only(path: Path, model: torch.nn.Module, prefix: str = "sonar_encoder.") -> None:
    """Load only sonar encoder weights by key prefix.

    Args:
        path: Full model checkpoint path.
        model: GeoCAM (or module containing ``sonar_encoder`` keys).
        prefix: State-dict prefix to keep (default GeoCAM sonar branch).
    """
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    filtered = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.sonar_encoder.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        pass
