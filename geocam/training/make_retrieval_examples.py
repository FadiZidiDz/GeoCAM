"""Generate qualitative retrieval figures from a trained GeoCAM checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from geocam.data.dataset import build_dataloaders, collate_sscam_batch
from geocam.models.geocam import GeoCAM
from geocam.training.pretrain import load_config
from geocam.utils.checkpoint import load_checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_raw_sonar(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    arr = np.log1p(np.clip(arr, 0.0, None))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def _load_raw_optical(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.uint8)


@torch.no_grad()
def _encode_tile(
    model: GeoCAM,
    sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    sonar, optical, weights, tile_int = sample
    batch = collate_sscam_batch([(sonar, optical, weights, tile_int)])
    sonar_b, optical_b, weights_b, pad_mask_b, _ = batch
    out = model(
        sonar_b.to(device),
        optical_b.to(device),
        weights_b.to(device),
        pad_mask_b.to(device),
        apply_cmmr_masking=False,
    )
    z_s = out["z_s"][0].float().cpu().numpy()
    z_o = out["z_o"][0].float().cpu().numpy()
    return z_s, z_o


def _make_retrieval_grid(
    sonar_imgs: List[np.ndarray],
    optical_imgs: List[np.ndarray],
    sims: np.ndarray,
    out_path: Path,
    topk: int,
    query_ids: List[int],
) -> None:
    n_q = len(query_ids)
    fig, axes = plt.subplots(n_q, topk + 1, figsize=(2.3 * (topk + 1), 2.1 * n_q))
    if n_q == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, qi in enumerate(query_ids):
        ax0 = axes[r, 0]
        ax0.imshow(sonar_imgs[qi], cmap="gray")
        ax0.set_title(f"Query sonar #{qi}", fontsize=9)
        ax0.axis("off")
        rank = np.argsort(-sims[qi])[:topk]
        for c, bj in enumerate(rank, start=1):
            ax = axes[r, c]
            ax.imshow(optical_imgs[bj])
            is_true = qi == bj
            ax.set_title(f"Top-{c}\nidx {bj}", fontsize=8, color=("green" if is_true else "red"))
            ax.axis("off")
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(2.0)
                sp.set_edgecolor("green" if is_true else "red")
    fig.suptitle("GeoCAM Cross-Modal Retrieval Examples", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_embedding_plot(z_s: np.ndarray, z_o: np.ndarray, out_path: Path, max_pairs: int = 200) -> None:
    n = min(len(z_s), len(z_o), max_pairs)
    xs = z_s[:n]
    xo = z_o[:n]
    both = np.concatenate([xs, xo], axis=0)
    # Try TSNE first; fallback to PCA if sklearn is unavailable.
    try:
        from sklearn.manifold import TSNE  # type: ignore

        emb = TSNE(n_components=2, perplexity=min(30, max(5, n // 3)), random_state=42).fit_transform(both)
    except Exception:
        c = both - both.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(c, full_matrices=False)
        emb = u[:, :2] * s[:2]
    es = emb[:n]
    eo = emb[n : 2 * n]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(es[:, 0], es[:, 1], c="#1f77b4", s=18, alpha=0.85, label="Sonar tiles")
    ax.scatter(eo[:, 0], eo[:, 1], c="#ff7f0e", s=18, alpha=0.85, label="Optical sets")
    for i in range(n):
        ax.plot([es[i, 0], eo[i, 0]], [es[i, 1], eo[i, 1]], c="gray", alpha=0.15, linewidth=0.8)
    ax.set_title("Embedding Space (t-SNE/PCA) with True Correspondence Links")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative GeoCAM retrieval figures")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config used for the trained checkpoint (e.g. geocam/configs/final_paper.yaml).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--output_dir", type=Path, default=Path("geocam_runs") / "paper_eval" / "figures")
    parser.add_argument("--max_tiles", type=int, default=300)
    parser.add_argument("--num_queries", type=int, default=6)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root_override", type=str, default="")
    args = parser.parse_args()

    _set_seed(args.seed)
    cfg: Dict[str, Any] = load_config(args.config)
    if args.data_root_override.strip():
        cfg["data"]["data_root"] = args.data_root_override.strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCAM(cfg).to(device)
    model.eval()
    load_checkpoint(args.checkpoint, model)

    _, val_loader = build_dataloaders(cfg)
    ds = val_loader.dataset
    if not hasattr(ds, "tile_ids"):
        raise RuntimeError("Expected SSCamDataset with tile_ids.")

    limit = min(int(args.max_tiles), len(ds))
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:limit]

    z_s_list: List[np.ndarray] = []
    z_o_list: List[np.ndarray] = []
    sonar_imgs: List[np.ndarray] = []
    optical_imgs: List[np.ndarray] = []

    for idx in indices:
        sample = ds[idx]
        z_s, z_o = _encode_tile(model, sample, device=device)
        z_s_list.append(z_s)
        z_o_list.append(z_o)

        tile_stem = ds.tile_ids[idx]
        sonar_path = ds.sonar_dir / f"{tile_stem}.npy"
        sonar_imgs.append(_load_raw_sonar(sonar_path))

        first_cam_stem = ds.correspondences[tile_stem][0]
        optical_path = ds.camera_dir / f"{first_cam_stem}.png"
        optical_imgs.append(_load_raw_optical(optical_path))

    z_s = np.asarray(z_s_list, dtype=np.float32)
    z_o = np.asarray(z_o_list, dtype=np.float32)
    sims = z_s @ z_o.T

    num_q = min(int(args.num_queries), len(indices))
    query_ids = random.sample(list(range(len(indices))), k=num_q)

    out_dir = args.output_dir
    _make_retrieval_grid(
        sonar_imgs=sonar_imgs,
        optical_imgs=optical_imgs,
        sims=sims,
        out_path=out_dir / "retrieval_examples_real.png",
        topk=int(args.topk),
        query_ids=query_ids,
    )
    _make_embedding_plot(z_s=z_s, z_o=z_o, out_path=out_dir / "embedding_correspondence_real.png")
    print(f"Wrote: {out_dir / 'retrieval_examples_real.png'}")
    print(f"Wrote: {out_dir / 'embedding_correspondence_real.png'}")


if __name__ == "__main__":
    main()
