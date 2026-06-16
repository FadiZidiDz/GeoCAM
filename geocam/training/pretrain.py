"""GeoCAM self-supervised pre-training loop with validation and retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from geocam.data.dataset import build_dataloaders
from geocam.losses.geo_infonce import GeoWeightedInfoNCE
from geocam.losses.reconstruction import PatchReconstructionLoss
from geocam.models.cmmr_decoder import CMMRDecoder
from geocam.models.geocam import GeoCAM
from geocam.utils.checkpoint import save_checkpoint
from geocam.utils.logger import SimpleLogger, set_seed
from geocam.utils.metrics import retrieval_metrics_from_embeddings


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file into nested dicts.

    Args:
        path: Path to ``default.yaml``.

    Returns:
        Parsed configuration.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_epoch_lrs(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    lr_encoder: float,
    lr_head: float,
) -> None:
    """Cosine decay with linear warmup, applied separately to param groups.

    Args:
        optimizer: AdamW with encoder group first, heads second.
        epoch: Current epoch (0-based).
        epochs: Total epochs.
        warmup_epochs: Warmup length in epochs.
        lr_encoder: Base LR for sonar encoder.
        lr_head: Base LR for auxiliary modules.
    """
    if epoch < warmup_epochs:
        scale = float(epoch + 1) / float(max(1, warmup_epochs))
    else:
        t = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        scale = 0.5 * (1.0 + math.cos(math.pi * t))
    optimizer.param_groups[0]["lr"] = lr_encoder * scale
    optimizer.param_groups[1]["lr"] = lr_head * scale


def saa_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy of SAA attention over optical slots.

    Args:
        attn: Tensor ``(B, 1, N)`` softmax weights.

    Returns:
        Scalar mean entropy.
    """
    w = attn.squeeze(1).clamp_min(1e-8)
    return (-(w * torch.log(w)).sum(dim=-1)).mean()


@torch.no_grad()
def evaluate(
    model: GeoCAM,
    loader: DataLoader,
    infonce: GeoWeightedInfoNCE,
    recon: PatchReconstructionLoss,
    lambda_cmmr: float,
    device: torch.device,
    fp16: bool,
    retrieval_ks: List[int],
    enable_cmmr_masking: bool,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Compute validation losses and retrieval metrics.

    Args:
        model: GeoCAM model.
        loader: Validation dataloader.
        infonce: Contrastive loss module.
        recon: Patch reconstruction loss.
        lambda_cmmr: Weight for reconstruction term.
        device: Torch device.
        fp16: Use autocast on CUDA.
        retrieval_ks: K values for Recall@K.
        enable_cmmr_masking: Whether to apply CMMR masking/loss in this stage.
        max_batches: If set, only run this many validation batches (faster smoke tests).

    Returns:
        ``(mean_total, mean_infonce, mean_cmmr, recall_dict)``.
    """
    model.eval()
    if len(loader) == 0:
        return 0.0, 0.0, 0.0, {f"R@{k}": 0.0 for k in retrieval_ks}
    totals = []
    ice = []
    cmm = []
    diag_sums: Dict[str, float] = {
        "pos_sim": 0.0,
        "neg_sim": 0.0,
        "pos_minus_neg": 0.0,
        "alignment": 0.0,
        "mean_weight": 0.0,
        "saa_entropy": 0.0,
    }
    z_s_all = []
    z_o_all = []
    pos_w_all = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        sonar, optical, weights, pad_mask, _ = batch
        sonar = sonar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda", dtype=torch.float16):
            out = model(sonar, optical, weights, pad_mask, apply_cmmr_masking=enable_cmmr_masking)
            target_patches = CMMRDecoder.patchify(sonar, model.patch_size)
            l_i, diag = infonce(out["z_s"], out["z_o"], out["pos_weights"])
            l_c = (
                recon(out["pred_patches"], target_patches, out["masked_ids"])
                if enable_cmmr_masking
                else torch.zeros((), device=device)
            )
            total = l_i + lambda_cmmr * l_c
        totals.append(total.item())
        ice.append(l_i.item())
        cmm.append(l_c.item())
        for k in diag_sums:
            if k == "saa_entropy":
                diag_sums[k] += float(saa_entropy(out["attn_weights"].float()).item())
            else:
                diag_sums[k] += float(diag[k])
        z_s_all.append(out["z_s"].float().cpu())
        z_o_all.append(out["z_o"].float().cpu())
        pos_w_all.append(out["pos_weights"].float().cpu())
    if not totals:
        return 0.0, 0.0, 0.0, {f"R@{k}": 0.0 for k in retrieval_ks}
    z_s_cat = torch.cat(z_s_all, dim=0)
    z_o_cat = torch.cat(z_o_all, dim=0)
    pos_w_cat = torch.cat(pos_w_all, dim=0)
    bsz = z_s_cat.size(0)
    recalls: Dict[str, float] = {f"R@{k}": 0.0 for k in retrieval_ks}
    if bsz > 1:
        z_s_cat = z_s_cat.to(device)
        z_o_cat = z_o_cat.to(device)
        pos_w_cat = pos_w_cat.to(device)
        s2o_metrics = retrieval_metrics_from_embeddings(
            z_s_cat,
            z_o_cat,
            retrieval_ks,
            positive_weights=pos_w_cat,
            prefix="s2o/",
        )
        o2s_metrics = retrieval_metrics_from_embeddings(
            z_o_cat,
            z_s_cat,
            retrieval_ks,
            positive_weights=pos_w_cat,
            prefix="o2s/",
        )
        global_s2o = retrieval_metrics_from_embeddings(
            z_s_cat,
            z_o_cat,
            retrieval_ks,
            positive_weights=pos_w_cat,
            prefix="global_s2o/",
        )
        global_o2s = retrieval_metrics_from_embeddings(
            z_o_cat,
            z_s_cat,
            retrieval_ks,
            positive_weights=pos_w_cat,
            prefix="global_o2s/",
        )
        recalls = {}
        recalls.update(s2o_metrics)
        recalls.update(o2s_metrics)
        recalls.update(global_s2o)
        recalls.update(global_o2s)
        # Keep backward-compatible summary keys for previous scripts.
        recalls.update(
            {
                f"R@{k}": float(
                    0.5 * (s2o_metrics.get(f"s2o/R@{k}", 0.0) + o2s_metrics.get(f"o2s/R@{k}", 0.0))
                )
                for k in retrieval_ks
            }
        )
    n_batches = float(len(totals))
    recalls.update({f"diag/{k}": (v / n_batches) for k, v in diag_sums.items()})
    return (
        float(sum(totals) / len(totals)),
        float(sum(ice) / len(ice)),
        float(sum(cmm) / len(cmm)),
        recalls,
    )


def train_one_epoch(
    model: GeoCAM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    infonce: GeoWeightedInfoNCE,
    recon: PatchReconstructionLoss,
    lambda_cmmr: float,
    device: torch.device,
    fp16: bool,
    scaler: torch.amp.GradScaler,
    clip_grad: float,
    log_every: int,
    logger: SimpleLogger,
    global_step: int,
    enable_cmmr_masking: bool,
    max_batches: Optional[int] = None,
) -> int:
    """Run one training epoch.

    Args:
        model: GeoCAM model.
        loader: Training dataloader.
        optimizer: AdamW optimizer.
        infonce: Contrastive loss.
        recon: CMMR loss.
        lambda_cmmr: CMMR weight.
        device: Device.
        fp16: Enable AMP on CUDA.
        scaler: Grad scaler.
        clip_grad: Max grad norm.
        log_every: Log interval in steps.
        logger: Logger.
        global_step: Starting step.
        enable_cmmr_masking: Whether to apply CMMR masking/loss in this stage.
        max_batches: If set, stop after this many optimizer steps (smoke tests).

    Returns:
        Next global step after the epoch.
    """
    model.train()
    step = global_step
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        sonar, optical, weights, pad_mask, _ = batch
        sonar = sonar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda", dtype=torch.float16):
            out = model(sonar, optical, weights, pad_mask, apply_cmmr_masking=enable_cmmr_masking)
            target_patches = CMMRDecoder.patchify(sonar, model.patch_size)
            l_i, diag = infonce(out["z_s"], out["z_o"], out["pos_weights"])
            l_c = (
                recon(out["pred_patches"], target_patches, out["masked_ids"])
                if enable_cmmr_masking
                else torch.zeros((), device=device)
            )
            loss = l_i + lambda_cmmr * l_c
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if batch_idx % log_every == 0:
            ent = saa_entropy(out["attn_weights"].float())
            logger.log(
                {
                    "train/step": step,
                    "train/loss": float(loss.item()),
                    "train/infonce": float(l_i.item()),
                    "train/cmmr": float(l_c.item()),
                    "train/saa_entropy": float(ent.item()),
                    "train/pos_sim": float(diag["pos_sim"]),
                    "train/neg_sim": float(diag["neg_sim"]),
                    "train/pos_minus_neg": float(diag["pos_minus_neg"]),
                    "train/alignment": float(diag["alignment"]),
                    "train/mean_weight": float(diag["mean_weight"]),
                    "train/lr_enc": optimizer.param_groups[0]["lr"],
                    "train/lr_head": optimizer.param_groups[1]["lr"],
                },
                step=step,
            )
    return step


def main() -> None:
    """CLI entry for GeoCAM pre-training (single GPU)."""
    parser = argparse.ArgumentParser(description="GeoCAM pre-training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Stop each epoch after this many train batches (for quick CPU/GPU smoke tests).",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Limit validation to this many batches (faster smoke eval).",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("geocam_runs") / "default")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = int(args.num_workers)

    seed = int(cfg["data"]["seed"])
    set_seed(seed)

    data_root = Path(cfg["data"]["data_root"])
    overlap_path = data_root / cfg["data"]["overlap_weights"]
    if not overlap_path.is_file():
        raise FileNotFoundError(
            f"Missing {overlap_path}. Run: python -m geocam.data.precompute_overlaps --data_root {data_root}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)

    model = GeoCAM(cfg).to(device)
    hard_negs = bool(cfg["training"].get("hard_negatives", False))
    hn_boost = float(cfg["training"].get("hard_negative_boost", 0.0))
    infonce = GeoWeightedInfoNCE(
        temperature=float(cfg["training"]["temperature"]),
        hard_negatives=hard_negs,
        hard_negative_boost=hn_boost,
    ).to(device)
    recon = PatchReconstructionLoss().to(device)

    head_params: List[nn.Parameter] = []
    head_params += list(model.saa.parameters())
    head_params += list(model.sonar_proj.parameters())
    head_params += list(model.optical_proj.parameters())
    head_params += list(model.cmmr_decoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": model.sonar_encoder.parameters(), "lr": float(cfg["training"]["lr_encoder"])},
            {"params": head_params, "lr": float(cfg["training"]["lr_head"])},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    epochs = int(cfg["training"]["epochs"])
    warmup = int(cfg["training"]["warmup_epochs"])
    fp16 = bool(cfg["training"]["fp16"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))
    lambda_cmmr = float(cfg["training"]["lambda_cmmr"])
    enable_cmmr_masking = bool(cfg["training"].get("enable_cmmr_masking", True))
    clip_grad = float(cfg["training"]["clip_grad"])
    log_every = int(cfg["logging"].get("log_every", 50))
    save_best_only = bool(cfg["logging"].get("save_best_only", True))
    save_last = bool(cfg["logging"].get("save_last", False))
    save_periodic_every = int(cfg["logging"].get("save_periodic_every", 10))
    retrieval_ks = [int(k) for k in cfg["eval"].get("retrieval_k", [1, 5])]
    if 10 not in retrieval_ks:
        retrieval_ks = sorted(set(retrieval_ks + [10]))

    logger = SimpleLogger(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    final_metrics: Dict[str, float] = {}
    epoch_rows: List[Dict[str, float]] = []
    global_step = 0
    low_entropy_streak = 0

    for epoch in range(epochs):
        set_epoch_lrs(
            optimizer,
            epoch,
            epochs,
            warmup,
            float(cfg["training"]["lr_encoder"]),
            float(cfg["training"]["lr_head"]),
        )
        global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            infonce,
            recon,
            lambda_cmmr,
            device,
            fp16,
            scaler,
            clip_grad,
            log_every,
            logger,
            global_step,
            enable_cmmr_masking,
            max_batches=args.max_train_batches,
        )
        v_total, v_i, v_c, recalls = evaluate(
            model,
            val_loader,
            infonce,
            recon,
            lambda_cmmr,
            device,
            fp16,
            retrieval_ks,
            enable_cmmr_masking,
            max_batches=args.max_val_batches,
        )
        logger.log(
            {
                "epoch": epoch,
                "val/loss": v_total,
                "val/infonce": v_i,
                "val/cmmr": v_c,
                **{f"val/{k}": v for k, v in recalls.items()},
            },
            step=global_step,
        )
        final_metrics = {
            "epoch": float(epoch),
            "val/loss": float(v_total),
            "val/infonce": float(v_i),
            "val/cmmr": float(v_c),
            **{f"val/{k}": float(v) for k, v in recalls.items()},
        }
        epoch_rows.append(dict(final_metrics))
        val_entropy = float(recalls.get("diag/saa_entropy", 0.0))
        if val_entropy < 0.5:
            low_entropy_streak += 1
        else:
            low_entropy_streak = 0
        if low_entropy_streak >= 5:
            print("Warning: SAA may have collapsed — check learning rate / regularization.")
        if v_total < best_val:
            best_val = v_total
            best_epoch = epoch
            best_metrics = {
                "val/loss": float(v_total),
                "val/infonce": float(v_i),
                "val/cmmr": float(v_c),
                **{f"val/{k}": float(v) for k, v in recalls.items()},
            }
            save_checkpoint(
                model,
                optimizer,
                epoch,
                v_total,
                args.output_dir / "checkpoint_best.pt",
                extra={"best_val": best_val},
            )
        if (not save_best_only) and (
            (save_periodic_every > 0 and (epoch + 1) % save_periodic_every == 0) or epoch == epochs - 1
        ):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                v_total,
                args.output_dir / f"checkpoint_epoch_{epoch+1:04d}.pt",
            )

    if save_last:
        save_checkpoint(
            model,
            optimizer,
            epochs - 1,
            best_val,
            args.output_dir / "checkpoint_last.pt",
        )

    # Write a persistent evaluation report for this run.
    report = {
        "output_dir": str(args.output_dir),
        "epochs": epochs,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "checkpoint_best": str(args.output_dir / "checkpoint_best.pt"),
        "save_best_only": save_best_only,
        "save_last": save_last,
    }
    report_json = args.output_dir / "training_report.json"
    report_txt = args.output_dir / "training_report.txt"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with report_txt.open("w", encoding="utf-8") as f:
        f.write("GeoCAM Training Report\n")
        f.write("=====================\n")
        f.write(f"Output dir: {report['output_dir']}\n")
        f.write(f"Epochs: {report['epochs']}\n")
        f.write(f"Best epoch: {report['best_epoch']}\n")
        f.write(f"Best val/loss: {report['best_val_loss']:.6f}\n")
        f.write(f"Best checkpoint: {report['checkpoint_best']}\n\n")
        f.write("Best metrics:\n")
        for k, v in sorted(best_metrics.items()):
            f.write(f"- {k}: {v:.6f}\n")
        f.write("\nFinal metrics:\n")
        for k, v in sorted(final_metrics.items()):
            f.write(f"- {k}: {v:.6f}\n")
    if epoch_rows:
        metrics_csv = args.output_dir / "epoch_metrics.csv"
        all_cols = sorted({k for row in epoch_rows for k in row.keys()})
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            writer.writerows(epoch_rows)
    print(f"Wrote training reports: {report_json} and {report_txt}")


if __name__ == "__main__":
    main()
