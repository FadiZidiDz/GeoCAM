"""Seeding and optional Weights & Biases logging."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (CPU and CUDA if available).

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


class SimpleLogger:
    """Console (and optional W&B) logger."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialize logger from ``cfg['logging']``.

        Args:
            cfg: Full GeoCAM config dict.
        """
        self.cfg = cfg.get("logging", {})
        self.use_wandb = bool(self.cfg.get("use_wandb", False))
        self._run = None
        if self.use_wandb:
            import wandb

            self._run = wandb.init(
                project=str(self.cfg.get("project", "geocam")),
                config=cfg,
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Record scalars to console and optionally W&B.

        Args:
            metrics: Mapping of names to scalar values.
            step: Optional global step for charts.
        """
        parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        print(" | ".join(parts))
        if self._run is not None:
            import wandb

            wandb.log(metrics, step=step)
