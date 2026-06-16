"""SSS-CAM PyTorch dataset and dataloaders (tile-centric split)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from geocam.data.correspondences import load_normalized_correspondences
from geocam.data.transforms import (
    OpticalTransform,
    SonarTransform,
    default_optical_transform,
    default_sonar_transform,
)


def load_overlap_weights(path: Path) -> Dict[str, Dict[str, float]]:
    """Load nested overlap weight JSON.

    Args:
        path: Path to ``overlap_weights.json``.

    Returns:
        Dict mapping tile stem → camera stem → overlap ratio.
    """
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, float]] = json.load(f)
    return data


def train_val_tile_split(
    tile_ids: List[str],
    train_fraction: float,
    seed: int,
    sonar_csv: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Randomly split tile IDs into train/val (no image-level leakage).

    Args:
        tile_ids: Sorted or unsorted list of tile stems.
        train_fraction: Fraction of tiles for training.
        seed: RNG seed.
        sonar_csv: Optional metadata CSV for geographic stratification.

    Returns:
        ``(train_tiles, val_tiles)``.
    """
    rng = random.Random(seed)
    ids = list(tile_ids)
    if sonar_csv is not None and sonar_csv.is_file():
        try:
            s_df = pd.read_csv(sonar_csv)
            path_col = "Patch_File_Path"
            east_col = "Top_Left_East"
            if path_col in s_df.columns and east_col in s_df.columns:
                stem_to_east: Dict[str, float] = {}
                for _, row in s_df.iterrows():
                    stem = Path(str(row[path_col])).stem
                    stem_to_east[stem] = float(row[east_col])
                ids_with_meta = [t for t in ids if t in stem_to_east]
                if len(ids_with_meta) >= 10:
                    vals = np.array([stem_to_east[t] for t in ids_with_meta], dtype=np.float64)
                    q_edges = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
                    bins: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: [], 4: []}
                    for t in ids:
                        if t not in stem_to_east:
                            continue
                        v = stem_to_east[t]
                        b = int(np.searchsorted(q_edges, v, side="right"))
                        bins[b].append(t)
                    train: List[str] = []
                    val: List[str] = []
                    for b in range(5):
                        group = bins[b]
                        if not group:
                            continue
                        rng.shuffle(group)
                        n = len(group)
                        if n == 1:
                            train.extend(group)
                            continue
                        n_train = int(n * train_fraction)
                        n_train = max(1, min(n_train, n - 1))
                        train.extend(group[:n_train])
                        val.extend(group[n_train:])
                    if train and val:
                        return train, val
        except Exception:
            # Fall back to random split if metadata stratification fails.
            pass
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return [], []
    if n == 1:
        return ids, []
    n_train = int(n * train_fraction)
    n_train = max(1, min(n_train, n - 1))
    train = ids[:n_train]
    val = ids[n_train:]
    return train, val


def _sample_camera_stems(
    cam_stems: List[str],
    weights: List[float],
    k: int,
    use_overlap_sampling: bool,
) -> Tuple[List[str], List[float]]:
    """Sample up to ``k`` camera stems (optionally overlap-weighted, without replacement)."""
    if len(cam_stems) <= k:
        return list(cam_stems), list(weights)
    if not use_overlap_sampling:
        idx = random.sample(range(len(cam_stems)), k)
    else:
        w = torch.tensor(weights, dtype=torch.float64)
        w = torch.clamp(w, min=1e-8)
        idx_t = torch.multinomial(w, k, replacement=False)
        idx = idx_t.tolist()
    chosen = [cam_stems[i] for i in idx]
    chosen_w = [weights[i] for i in idx]
    return chosen, chosen_w


class SSCamDataset(Dataset):
    """One sample = one sonar tile + a subset of matched optical images."""

    def __init__(
        self,
        data_root: Path,
        correspondences: Dict[str, List[str]],
        overlap_weights: Dict[str, Dict[str, float]],
        tile_ids: List[str],
        sonar_subdir: str,
        camera_subdir: str,
        max_optical: int,
        sonar_transform: Optional[SonarTransform] = None,
        optical_transform: Optional[OpticalTransform] = None,
        optical_size: int = 224,
        use_overlap_sampling: bool = False,
    ) -> None:
        """Build a tile-indexed dataset.

        Args:
            data_root: Dataset root.
            correspondences: tile stem → camera stems.
            overlap_weights: tile stem → camera stem → overlap ratio.
            tile_ids: Tiles to include (train or val list).
            sonar_subdir: Relative path to ``*.npy`` tiles.
            camera_subdir: Relative path to ``*.png`` images.
            max_optical: Maximum optical views sampled per tile.
            sonar_transform: Optional sonar transforms.
            optical_transform: Optional optical transforms.
            optical_size: Resize target for RGB (used if transforms None for eval default).
            use_overlap_sampling: If True, bias optical subset toward high-overlap frames.
        """
        super().__init__()
        self.data_root = data_root
        self.sonar_dir = data_root / sonar_subdir
        self.camera_dir = data_root / camera_subdir
        self.correspondences = correspondences
        self.overlap_weights = overlap_weights
        self.tile_ids = list(tile_ids)
        self.max_optical = max_optical
        self.sonar_transform = sonar_transform
        self.optical_transform = optical_transform
        self.optical_size = optical_size
        self.use_overlap_sampling = use_overlap_sampling

    def __len__(self) -> int:
        """Number of tiles in this split."""
        return len(self.tile_ids)

    def _load_sonar(self, tile_stem: str) -> torch.Tensor:
        """Load ``(1, H, W)`` float sonar tile from NPY."""
        path = self.sonar_dir / f"{tile_stem}.npy"
        arr = np.load(path).astype(np.float32)
        t = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t

    def _load_optical(self, cam_stem: str) -> torch.Tensor:
        """Load RGB image as ``(3, H, W)`` float in [0, 1]."""
        path = self.camera_dir / f"{cam_stem}.png"
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return sonar, optical stack, overlap weights, and numeric tile id.

        Args:
            index: Dataset index.

        Returns:
            ``(sonar, opticals, weights, tile_int)`` with shapes
            ``(1,H,W)``, ``(N,3,S,S)``, ``(N,)``, scalar int tile id.
        """
        tile_stem = self.tile_ids[index]
        cam_stems = list(self.correspondences[tile_stem])
        wmap = self.overlap_weights.get(tile_stem, {})
        # If overlap weights are missing for a correspondence, fall back to a neutral
        # value rather than 0.0 (which would strongly penalize the image in SAA geo-bias).
        weights_all = [float(wmap.get(c, 1.0)) for c in cam_stems]
        chosen, w_chosen = _sample_camera_stems(
            cam_stems,
            weights_all,
            self.max_optical,
            self.use_overlap_sampling,
        )

        sonar = self._load_sonar(tile_stem)
        if self.sonar_transform is not None:
            sonar = self.sonar_transform(sonar)

        optical_tensors: List[torch.Tensor] = []
        for cs in chosen:
            o = self._load_optical(cs)
            if self.optical_transform is not None:
                o = self.optical_transform(o)
            else:
                o = torch.nn.functional.interpolate(
                    o.unsqueeze(0),
                    size=(self.optical_size, self.optical_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            optical_tensors.append(o)
        optical_stack = torch.stack(optical_tensors, dim=0)
        weight_tensor = torch.tensor(w_chosen, dtype=torch.float32)
        tile_int = int(tile_stem) if tile_stem.isdigit() else index
        return sonar, optical_stack, weight_tensor, tile_int


def collate_sscam_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable ``N`` optical views; mask padded slots.

    Args:
        batch: List of dataset items.

    Returns:
        ``sonar (B,1,H,W)``, ``optical (B,Nmax,3,S,S)``, ``weights (B,Nmax)``,
        ``padding_mask (B,Nmax)`` True where padded, ``tile_ids (B,)``.
    """
    sonars = [b[0] for b in batch]
    opts = [b[1] for b in batch]
    ws = [b[2] for b in batch]
    tile_ids = torch.tensor([b[3] for b in batch], dtype=torch.long)

    bsz = len(batch)
    nmax = max(o.shape[0] for o in opts)
    _, c, h, w = opts[0].shape

    optical = torch.zeros(bsz, nmax, c, h, w, dtype=opts[0].dtype)
    weights = torch.zeros(bsz, nmax, dtype=torch.float32)
    padding_mask = torch.ones(bsz, nmax, dtype=torch.bool)

    for i, o in enumerate(opts):
        n = o.shape[0]
        optical[i, :n] = o
        weights[i, :n] = ws[i]
        padding_mask[i, :n] = False

    sonar_batch = torch.stack(sonars, dim=0)
    return sonar_batch, optical, weights, padding_mask, tile_ids


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders from a nested config dict.

    Args:
        cfg: Full GeoCAM config (must include ``data`` section).

    Returns:
        ``(train_loader, val_loader)``.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    data_root = Path(data_cfg["data_root"])
    corr_path = data_root / data_cfg["correspondences"]
    overlap_path = data_root / data_cfg["overlap_weights"]

    corr, all_tiles = load_normalized_correspondences(
        corr_path,
        data_root,
        data_cfg["sonar_subdir"],
        data_cfg["camera_subdir"],
        verify_files=True,
    )
    overlaps = load_overlap_weights(overlap_path)

    ext = data_cfg.get("extension", {}) or {}
    use_overlap_sampling = str(ext.get("name", "none")) == "overlap_sampling"

    train_tiles, val_tiles = train_val_tile_split(
        all_tiles,
        float(data_cfg["train_split"]),
        int(data_cfg["seed"]),
        data_root / data_cfg["sonar_csv"] if "sonar_csv" in data_cfg else None,
    )

    img_size = int(cfg["model"]["sonar"]["img_size"])
    train_ds = SSCamDataset(
        data_root=data_root,
        correspondences=corr,
        overlap_weights=overlaps,
        tile_ids=train_tiles,
        sonar_subdir=data_cfg["sonar_subdir"],
        camera_subdir=data_cfg["camera_subdir"],
        max_optical=int(data_cfg["max_optical_per_tile"]),
        sonar_transform=default_sonar_transform(augment=True, img_size=img_size),
        optical_transform=default_optical_transform(augment=True, out_size=224),
        use_overlap_sampling=use_overlap_sampling,
    )
    val_ds = SSCamDataset(
        data_root=data_root,
        correspondences=corr,
        overlap_weights=overlaps,
        tile_ids=val_tiles,
        sonar_subdir=data_cfg["sonar_subdir"],
        camera_subdir=data_cfg["camera_subdir"],
        max_optical=int(data_cfg["max_optical_per_tile"]),
        sonar_transform=default_sonar_transform(augment=False, img_size=img_size),
        optical_transform=default_optical_transform(augment=False, out_size=224),
        use_overlap_sampling=False,
    )

    num_workers = int(train_cfg.get("num_workers", 4))
    bs = int(train_cfg["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sscam_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sscam_batch,
        drop_last=False,
    )
    return train_loader, val_loader
