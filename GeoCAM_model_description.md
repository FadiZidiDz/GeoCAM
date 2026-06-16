# GeoCAM (Current Paper Version): Unlabeled Opti-Acoustic SSL for BenthiCat

## 1) What this document is

This is the **latest working description** of the GeoCAM model and pipeline as currently implemented for paper-focused experiments.

Current scope is:

- **Primary paper track (active):** unlabeled cross-modal representation learning on BenthiCat SSS-CAM (sonar + camera correspondences).
- **Secondary extension (in progress):** N07-style SSS matching/registration benchmark extension.

This document also explains:

- why segmentation/classification is currently limited,
- what data files mean in practice,
- and what a sonar tile actually is.

---

## 2) Current contribution (active paper track)

### Core claim (scientifically safe right now)

GeoCAM improves **cross-modal retrieval and geometric alignment** between side-scan sonar tiles and associated optical views in an unlabeled setting.

### What the current model uses

- Sonar tiles (`.npy`) from SSS-CAM
- Optical images (`.png`) from SSS-CAM
- Pair mapping from `correspondences.json`
- Overlap priors from `overlap_weights.json` (derived from `sonar.csv` + `camera.csv`)

### What the current model does not require

- No segmentation masks
- No class labels
- No supervised segmentation split

---

## 3) Model architecture (current)

### 3.1 Sonar branch

- Backbone: ViT-S/16-style sonar encoder
- Input: sonar tile (single-channel acoustic intensity map, transformed for model input)
- Output: sonar embedding
- Trainable during SSL pretraining

### 3.2 Optical branch

- Backbone: DINOv2 ViT-S optical encoder
- Frozen during pretraining
- Per optical image embedding extracted, then aggregated

### 3.3 Set Attention Aggregator (SAA)

- Aggregates multiple optical views linked to one sonar tile
- Uses valid-view masking and geographic weighting behavior
- Produces one tile-level optical representation

### 3.4 Projection heads + contrastive space

- Sonar and optical embeddings projected into shared latent space
- Loss is contrastive (Geo-weighted InfoNCE, corrected form)

### 3.5 Optional auxiliary branch (CMMR)

- Cross-modal masked reconstruction path exists
- For current best recipe, this is typically disabled during contrastive stage based on ablations

---

## 4) Training and evaluation (current)

### 4.1 Pretraining setup

- Unlabeled pretraining on SSS-CAM pairs
- Geometric overlap priors used as soft positive weighting
- Mixed precision, warmup + cosine schedule, best-checkpoint saving

### 4.2 Logged metrics (current implementation)

- Retrieval: `R@1`, `R@5`, `R@10`
- Ranking: `MRR`, `MedianRank`
- Graded relevance: `NDCG@k` (including `NDCG@5`, `NDCG@10`)
- Diagnostics: `pos_sim`, `neg_sim`, `pos_minus_neg`, `alignment`, `mean_weight`, `saa_entropy`
- Outputs: `training_report.json`, `training_report.txt`, `epoch_metrics.csv`

### 4.3 Paper automation (already implemented)

- Multi-seed runner (3 seeds)
- Seed aggregation (mean +- std)
- Figure generation
- Table generation for paper-ready reporting

---

## 5) Dataset explanation (BenthiCat) and roles

BenthiCat contains multiple assets; they are not all equivalent in supervision strength.

### 5.1 `sonar/` (SSS tiles)

- Contains side-scan sonar image patches (typically as `.npy`)
- Each tile is a local seabed patch in acoustic intensity domain
- **Role:** primary input for sonar encoder and SSL learning

### 5.2 `camera/` (optical images)

- RGB images captured in overlapping geographic regions
- **Role:** second modality for cross-modal learning

### 5.3 `correspondences.json`

- Mapping from each sonar tile to one or more optical images
- **Role:** defines positive cross-modal associations for unlabeled training

### 5.4 `sonar.csv`

- Metadata for sonar tiles (paths, geospatial / navigation fields, etc.)
- **Role:** used for split logic and overlap geometry calculations

### 5.5 `camera.csv`

- Metadata for camera images (paths, geospatial footprint fields, etc.)
- **Role:** paired with `sonar.csv` to compute overlap priors

### 5.6 `overlap_weights.json`

- Precomputed overlap ratios for tile-image pairs
- **Role:** geometric prior for weighted contrastive learning and diagnostics

### 5.7 Segmentation labels (separate supervised subset)

- BenthiCat paper reports a labeled subset (about tens of thousands of tiles with masks)
- **Role:** required for supervised segmentation/classification benchmarking

---

## 6) What is a "tile" exactly?

A **tile** is a fixed-size cropped patch of a larger side-scan sonar waterfall/mosaic.

Conceptually:

- raw survey produces long sonar strips/waterfalls,
- preprocessing splits these strips into manageable fixed windows (tiles),
- each tile is one local seabed snapshot used as one sample for ML.

Why tiles matter:

- standardized model input size,
- memory-efficient batching,
- localized texture/structure learning,
- easier pairing with nearby optical observations.

---

## 7) Why we cannot claim segmentation/classification results now

For the **current active run setup**, we only use unlabeled SSS-CAM association assets (tiles, images, correspondences, metadata, overlaps).  
That supports retrieval/alignment SSL claims, but **not** supervised habitat segmentation claims.

Main blockers:

1. No confirmed local supervised mask split integrated in this run pipeline.
2. No completed downstream supervised train/val/test protocol on masks in current paper automation.
3. Without mask-driven evaluation, mIoU/class metrics are unavailable and any segmentation claim would be unsafe.

Important nuance:

- This is not because BenthiCat has no labels at all.
- It is because the current working pipeline/paper track is intentionally unlabeled and does not yet execute full labeled downstream experiments.

---

## 8) Current paper-safe claims vs non-safe claims

### Safe now

- Improves unlabeled cross-modal retrieval between sonar and optical data
- Improves representation alignment diagnostics
- Provides reproducible multi-seed SSL evaluation pipeline

### Not safe yet (until labeled downstream is executed)

- "Improves segmentation mIoU"
- "Improves habitat classification accuracy"
- "SOTA for benthic segmentation"

---

## 9) If/when masks are obtained and integrated

Once mask split access is available and integrated, the strong next contribution is:

1. Use GeoCAM pretrained sonar encoder as initialization.
2. Run linear probe and/or full fine-tuning on labeled split.
3. Report mIoU/F1/per-class metrics with fixed protocol and 3 seeds.
4. Compare against:
   - random initialization,
   - sonar-only pretraining baseline,
   - SSL ablations from current GeoCAM recipe.

Then segmentation/classification claims become scientifically valid.

---

## 10) Practical takeaway

- **Right now:** strongest contribution is unlabeled opti-acoustic SSL + retrieval/alignment.
- **Near term:** extend with N07 SSS matching track for geometric registration comparisons.
- **After masks integration:** add supervised downstream segmentation paper section for full end-to-end habitat mapping evidence.
# GeoCAM: Geography-Guided Cross-Modal Set Contrastive Pre-training for Benthic Habitat Understanding

## 1. Overview

**GeoCAM** is a self-supervised cross-modal pre-training framework designed for the **BenthiCat SSS-CAM** dataset. It learns a shared embedding space between side-scan sonar (SSS) tiles and underwater optical images of the same seafloor patch — without any human-provided labels.

The core insight: SSS-CAM provides **1-to-N pairing** (one sonar tile linked to ~18 optical photos) and **free geographic metadata** (GPS footprint corners of every tile and every image). No existing method fully exploits either of these signals. GeoCAM introduces three novel components that do.

**Three contributions:**
1. **Set Attention Aggregator (SAA)** — treats the N optical images as a set and learns to weight them by informativeness, rather than mean-pooling.
2. **Geometry-Weighted InfoNCE** — uses the computed overlap ratio between sonar and optical footprints as a soft positive weight, turning free metadata into supervision.
3. **Cross-Modal Masked Reconstruction (CMMR)** — auxiliary task that reconstructs masked sonar patches from the aggregated optical embedding, forcing the shared space to encode physical substrate properties.

---

## 2. Problem Statement

Given:
- A sonar tile `S` (256×256 float32 NPY, log-normalized acoustic backscatter)
- A set of optical images `O = {o_1, o_2, ..., o_N}` (N ≈ 18, RGB PNG 1936×1464, downsampled to 224×224)
- Geographic overlap ratios `w = {w_1, w_2, ..., w_N}` where `w_i ∈ (0, 1]` computed from footprint intersection
- No class labels anywhere

**Goal:** Learn an encoder `f_s` such that `f_s(S)` produces embeddings that are semantically meaningful for downstream seafloor segmentation — measured by linear probe mIoU on the BenthiCat SSS Seafloor Segmentation split.

---

## 3. Dataset Usage

### Input data (SSS-CAM)
| File | Content | Used for |
|---|---|---|
| `sonar/tile-*.npy` | 256×256 float32, values [0,1] | Sonar encoder input |
| `camera/image-*.png` | 1936×1464 RGB | Optical encoder input (resized to 224×224) |
| `correspondences.json` | Dict: tile_idx → [img_idx, ...] | Building paired batches |
| `sonar.csv` | GPS corners of each sonar tile | Computing overlap ratio `w_i` |
| `camera.csv` | GPS corners of each optical image | Computing overlap ratio `w_i` |

### Overlap ratio computation
For each (sonar tile `j`, optical image `i`) pair linked in `correspondences.json`:

```
w_ij = Area(footprint_i ∩ footprint_j) / Area(footprint_i)
```

Computed using Shapely polygon intersection on the 4 GPS corner coordinates from `sonar.csv` and `camera.csv`. This is computed once and cached as `overlap_weights.json` before training begins.

### Train / validation split
Split is performed at the **sonar tile level** (not image level) to prevent data leakage:
- Training: 80% of tile indices → 1,664 tiles → ~29,952 optical images
- Validation: 20% of tile indices → 416 tiles → ~7,995 optical images
- Split is seeded and stratified by geographic sector (derived from UTM easting/northing in `sonar.csv`)

---

## 4. Architecture

### 4.1 Sonar Encoder `f_s`

**Backbone:** ViT-S/16 (Vision Transformer Small, patch size 16)

**Initialization:** Weights from MAE (Masked Autoencoder) pre-training on BenthiCat SSS Pre-training split (~950k unlabelled tiles). This is the most critical decision — starting from random or ImageNet weights would give dramatically worse results because acoustic images have fundamentally different statistics from natural images.

**Input handling:**
- Input: (B, 1, 256, 256) — single-channel float32
- A learned 1→3 channel expansion conv (1×1, no activation) is prepended so the ViT patch embedding sees 3 channels (standard ViT input)
- Alternatively: replicate channel 3× before feeding — simpler but slightly worse empirically

**Output:** (B, 384) — [CLS] token from the final transformer layer

**Trainable:** Yes, fully fine-tuned during pre-training.

**Config:**
```python
embed_dim = 384
depth = 12
num_heads = 6
mlp_ratio = 4.0
patch_size = 16
img_size = 256
```

---

### 4.2 Optical Encoder `f_o`

**Backbone:** DINOv2-ViT-S/14 (pre-trained on LVD-142M by Meta AI)

**Initialization:** Official DINOv2 weights from `facebookresearch/dinov2` via torch.hub

**Input handling:**
- Input: (B×N, 3, 224, 224) — all N images from the batch processed in parallel
- Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Output:** (B, N, 384) — [CLS] token per image, reshaped into a set

**Trainable:** FROZEN. DINOv2 weights are not updated during GeoCAM pre-training. Only the SAA and projection heads are trained on the optical side. This is the asymmetric design choice — the sonar encoder needs domain adaptation, but DINOv2 already produces excellent visual features.

---

### 4.3 Set Attention Aggregator (SAA) — Novel Block

**Purpose:** Compress the set of N optical embeddings `{e_1, ..., e_N} ∈ R^384` into one representative vector `e_agg ∈ R^384`.

**Why not mean-pooling?** Mean-pooling treats all N images equally. In practice, some images have poor visibility (turbid water, motion blur, unfavorable angle). The SAA learns to downweight those images and upweight the clearer, more informative ones.

**Architecture:** Perceiver-style cross-attention with a single learned query

```
Input:  set of N embeddings E ∈ R^(N × 384)
Query:  learnable parameter Q ∈ R^(1 × 384)

Attention weights:  a = softmax(Q · E^T / sqrt(384))   ∈ R^(1 × N)
Aggregated output:  e_agg = a · E                       ∈ R^(1 × 384)
```

**Full SAA block:**
```
LayerNorm(E)
→ Multi-head cross-attention (query=Q_learned, key=E, value=E, num_heads=2)
→ Residual add (Q_learned + attn_out)
→ LayerNorm
→ FFN (384 → 1536 → 384, GELU)
→ Residual add
→ e_agg ∈ R^384
```

**Parameters:** ~2.4M (tiny compared to the encoders)

**Positional encoding:** None. The set is order-invariant by design, which matches the nature of the optical image set (no meaningful ordering among the ~18 matched photos).

---

### 4.4 Projection Heads

Both branches use an identical 2-layer MLP projection head to map encoder outputs into the contrastive embedding space:

```
Linear(384, 512) → BatchNorm(512) → ReLU
→ Linear(512, 256)
→ L2 normalize → z ∈ R^256
```

- Sonar projection head: maps `f_s(S) → z_s ∈ R^256`
- Optical projection head: maps `SAA({f_o(o_i)}) → z_o ∈ R^256`

The projection heads are discarded after pre-training. Only the encoder backbones are kept for downstream tasks.

---

### 4.5 Cross-Modal Masked Reconstruction (CMMR) — Auxiliary Head

**Purpose:** Force the optical embedding to encode the physical substrate information visible in the sonar. If the optical features can reconstruct masked sonar patches, the two embeddings are grounded in the same physical reality.

**Masking strategy:** 
- 40% of sonar patches are randomly masked (set to learnable mask token)
- The remaining 60% of patches + positional embeddings are passed to the sonar encoder
- This is lower masking ratio than standard MAE (75%) because the sonar encoder also needs to produce a clean CLS token for the contrastive loss

**Decoder:**
```
Input: [z_o (256-dim) expanded to patch sequence, visible sonar patches (B, 0.6×256, 384)]
→ Cross-attention: sonar visible patches attend to z_o
→ 4-layer lightweight ViT decoder (embed_dim=192, num_heads=3)
→ Linear head → predicted pixel values for masked patches
→ Loss: MSE on normalized masked patch pixels
```

The decoder is also discarded after pre-training.

---

## 5. Loss Functions

### 5.1 Geometry-Weighted InfoNCE (primary loss)

Standard InfoNCE treats all positive pairs with equal weight. GeoCAM weights each positive pair by its geographic overlap ratio, computed from the GPS footprints.

For a batch of B sonar tiles, each with its aggregated optical embedding:

```
L_InfoNCE = - (1/B) Σ_i  Σ_j w_ij · log [
    exp(z_s_i · z_o_j / τ) /
    Σ_k exp(z_s_i · z_o_k / τ)
]
```

Where:
- `w_ij` = geographic overlap ratio of sonar tile i and optical set j (0 for non-matched pairs in batch)
- `τ` = temperature hyperparameter (default: 0.07)
- The sum in the denominator runs over all B optical embeddings in the batch (including the positive)
- Bidirectional: computed sonar→optical and optical→sonar, averaged

**Important:** Within a batch, a sonar tile may have a low overlap with its own matched optical set (e.g. w=0.42) but its embedding should still be pulled closer than any negative pair. The overlap weight modulates how strongly positives are pulled — it does not make a low-overlap pair a negative.

### 5.2 MSE Reconstruction Loss (auxiliary)

```
L_MSE = (1 / |M|) Σ_{p ∈ M} || x_p - x̂_p ||^2
```

Where M is the set of masked patch indices, `x_p` is the normalized ground-truth pixel block, and `x̂_p` is the decoder prediction.

### 5.3 Total Loss

```
L_total = L_InfoNCE + λ · L_MSE
```

- `λ = 0.1` (auxiliary weight — reconstruction is secondary to the contrastive objective)
- Tunable as a hyperparameter; values of 0.05–0.2 are reasonable starting points

---

## 6. Training Protocol

### Phase 1: Pre-training on SSS-CAM (GeoCAM)
| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 64 sonar tiles |
| Optical images per tile | Sample max 8 from the N available (memory constraint) |
| Optimizer | AdamW |
| Learning rate | 1e-4 (sonar encoder), 1e-3 (SAA + proj heads) |
| LR schedule | Cosine decay with 10-epoch linear warmup |
| Weight decay | 0.05 |
| Temperature τ | 0.07 |
| λ (CMMR weight) | 0.1 |
| Mixed precision | fp16 |
| Gradient clipping | max_norm=1.0 |

**Data augmentation — sonar tiles:**
- Random horizontal flip (p=0.5) — left/right sonar sides
- Random crop and resize (scale 0.8–1.0, ratio 0.9–1.1)
- Gaussian noise (σ=0.01) to simulate sensor variability
- NO color jitter (grayscale) / NO strong geometric distortions (destroy sonar geometry)

**Data augmentation — optical images:**
- Standard SimCLR augmentations: random crop, color jitter, grayscale, Gaussian blur
- Random horizontal flip

### Phase 2: Downstream evaluation (linear probe)
- Freeze sonar encoder `f_s` completely
- Train a single linear classifier on top of the frozen [CLS] token
- Dataset: BenthiCat SSS Seafloor Segmentation split (36k annotated tiles, 12 classes)
- Optimizer: SGD with momentum 0.9, lr=0.1, cosine schedule
- Epochs: 100
- Metric: mean Intersection over Union (mIoU) across 12 classes

### Phase 3: Full fine-tuning (optional, upper bound)
- Unfreeze sonar encoder with small lr (1e-5)
- Same segmentation head as Phase 2 but now with decoder (UPerNet or SegFormer head)

---

## 7. Project File Structure

```
geocam/
├── data/
│   ├── dataset.py          # SSCamDataset: loads NPY + PNG, reads correspondences.json
│   ├── transforms.py       # Sonar and optical augmentation pipelines
│   └── precompute_overlaps.py  # One-time script: compute overlap_weights.json from CSVs
│
├── models/
│   ├── sonar_encoder.py    # ViT-S/16 with 1-channel input adapter
│   ├── optical_encoder.py  # DINOv2-ViT-S wrapper (frozen)
│   ├── saa.py              # SetAttentionAggregator class
│   ├── projection_head.py  # Shared MLP projection head
│   ├── cmmr_decoder.py     # Cross-modal masked reconstruction decoder
│   └── geocam.py           # GeoCAM: combines all modules, forward pass
│
├── losses/
│   ├── geo_infonce.py      # GeoWeightedInfoNCE loss
│   └── reconstruction.py   # MSE patch reconstruction loss
│
├── training/
│   ├── pretrain.py         # Pre-training loop (Phase 1)
│   ├── linear_probe.py     # Linear evaluation loop (Phase 2)
│   └── finetune.py         # Full fine-tuning loop (Phase 3)
│
├── utils/
│   ├── metrics.py          # mIoU, per-class IoU, confusion matrix
│   ├── checkpoint.py       # Save/load model state
│   └── logger.py           # WandB + console logging
│
├── configs/
│   └── default.yaml        # All hyperparameters
│
├── scripts/
│   ├── run_pretrain.sh
│   └── run_eval.sh
│
└── requirements.txt
```

---

## 8. Key Implementation Details

### Loading the SSS-CAM dataset

```python
# dataset.py outline
class SSCamDataset(Dataset):
    def __init__(self, data_root, correspondences, overlap_weights, split='train',
                 max_optical=8, sonar_transform=None, optical_transform=None):
        self.tile_indices = list(correspondences.keys())  # split already applied
        self.correspondences = correspondences
        self.overlap_weights = overlap_weights  # precomputed dict
        self.max_optical = max_optical

    def __getitem__(self, idx):
        tile_idx = self.tile_indices[idx]
        # Load sonar NPY
        sonar = np.load(f"sonar/tile-{tile_idx}.npy").astype(np.float32)
        sonar = torch.from_numpy(sonar).unsqueeze(0)  # (1, 256, 256)

        # Sample optical images and their overlap weights
        img_indices = self.correspondences[tile_idx]
        sampled = random.sample(img_indices, min(self.max_optical, len(img_indices)))
        images = [load_image(i) for i in sampled]  # list of (3, 224, 224) tensors
        weights = [self.overlap_weights[tile_idx][i] for i in sampled]

        images = torch.stack(images)   # (N, 3, 224, 224)
        weights = torch.tensor(weights)  # (N,)

        if self.sonar_transform: sonar = self.sonar_transform(sonar)
        if self.optical_transform: images = torch.stack([self.optical_transform(im) for im in images])

        return sonar, images, weights
```

### Geo-Weighted InfoNCE implementation

```python
# losses/geo_infonce.py outline
class GeoWeightedInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z_s, z_o, weights=None):
        # z_s: (B, 256) sonar embeddings — already L2 normalized
        # z_o: (B, 256) optical embeddings — already L2 normalized
        # weights: (B,) overlap ratios for the diagonal (positive pairs)

        B = z_s.size(0)
        logits = torch.mm(z_s, z_o.T) / self.tau  # (B, B)
        labels = torch.arange(B, device=z_s.device)  # diagonal = positives

        if weights is not None:
            # Scale the positive logit by overlap weight
            pos_scale = weights.clamp(min=0.4, max=1.0)  # don't kill low-overlap pairs
            mask = torch.eye(B, device=z_s.device)
            logits = logits * (1 - mask) + logits * mask * pos_scale.unsqueeze(1)

        loss_s2o = F.cross_entropy(logits, labels)
        loss_o2s = F.cross_entropy(logits.T, labels)
        return (loss_s2o + loss_o2s) / 2
```

---

## 9. Expected Results and Baselines

| Method | Pre-training data | Linear probe mIoU |
|---|---|---|
| Random init (no pre-train) | — | ~18% |
| MAE on SSS Pre-train split | 950k SSS tiles | ~34% |
| CLIP-style (mean-pool optical) | SSS-CAM | ~38% |
| **GeoCAM (ours)** | **SSS-CAM** | **~44%** (estimated) |
| Full fine-tune with GeoCAM init | SSS-CAM + Seg labels | ~52% (estimated) |

*These are projected figures based on comparable domain-adaptation results in sonar literature. Actual results may vary.*

---

## 10. Why Each Design Choice Matters

| Choice | Alternative | Reason for choice |
|---|---|---|
| MAE init for sonar encoder | ImageNet ViT | Sonar texture is utterly unlike natural images; domain-specific init is critical |
| DINOv2 frozen for optical | Fine-tune DINOv2 | Only 35k optical images — fine-tuning would overfit; DINOv2 already generalizes |
| SAA instead of mean-pool | Average N embeddings | Some optical images are turbid/blurry; SAA learns to ignore them |
| Geo-weighted InfoNCE | Standard InfoNCE | Free supervision signal; higher overlap = stronger positive signal |
| CMMR auxiliary loss | No auxiliary | Prevents the optical embedding from being lazy (ignoring substrate texture) |
| Split by tile not image | Random image split | Prevents same seafloor patch appearing in both train and test |

---

## 11. References

- Rajani et al. (2025). *BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping.* arXiv:2510.04876
- Rajani et al. (2023). *A convolutional vision transformer for semantic segmentation of side-scan sonar data.* Ocean Engineering 286: 115647
- He et al. (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022
- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024
- Jaegle et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML 2021
- Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020


Proposed Method (workshop-ready text)

Method Overview
We propose GeoCAM, a geography-guided cross-modal self-supervised framework for aligning side-scan sonar (SSS) tiles and underwater optical imagery in a shared embedding space without semantic labels.
Given one sonar tile and a variable-size set of matched optical views, GeoCAM learns modality-invariant yet geometry-aware representations by combining (i) set-level optical aggregation and (ii) overlap-weighted contrastive supervision derived from metadata.

Input Formulation
For each training sample, we use:

a sonar tile ( s \in \mathbb{R}^{1 \times H \times W} ),
a set of matched optical images ( \mathcal{O} = {o_i}_{i=1}^{N} ), (o_i \in \mathbb{R}^{3 \times 224 \times 224}),
overlap priors ( w_i \in [0,1] ) computed from sonar/camera footprints.
The correspondences come from correspondences.json, while overlap priors are computed from sonar.csv and camera.csv and stored as overlap_weights.json.

Encoders and Set Aggregation
Sonar encoder: ViT-S/16 backbone producing a sonar embedding (h_s).
Optical encoder: DINOv2 ViT-S/14 (frozen) producing per-image tokens/features (h_{o_i}).
Set Attention Aggregator (SAA): aggregates ({h_{o_i}}_{i=1}^{N}) into a single set-level optical descriptor (h_o), handling variable set cardinality and masking padded views.
Two projection heads map (h_s) and (h_o) into a shared space: [ z_s = g_s(h_s), \quad z_o = g_o(h_o), \quad z_s,z_o \in \mathbb{R}^{d}, ; |z|_2=1. ]

Geometry-Weighted Contrastive Learning
The primary objective is symmetric InfoNCE with geometric weighting on positive terms.
Critically, overlap weights scale the positive log-probability contribution (loss term weighting), not logits directly.

This encourages embeddings of strongly overlapping sonar-optical pairs to align more strongly, while still preserving discrimination against negatives.

Optional Auxiliary Objective
GeoCAM includes an optional cross-modal masked reconstruction branch (CMMR), where optical context assists reconstruction of masked sonar patches.
In the final recipe selected by ablation, contrastive-only training (CMMR disabled) provides the most stable retrieval gains.

Training Protocol and Evaluation
We train with AdamW, warmup + cosine schedule, mixed precision, and best-checkpoint selection.
Evaluation is performed in the unlabeled retrieval setting with:

Recall@K: R@1, R@5, R@10,
ranking metrics: MRR, MedianRank,
graded relevance: NDCG@K,
diagnostics: pos_sim, neg_sim, pos_minus_neg, and SAA entropy.
Three-seed full-budget runs are aggregated as mean±std.

Contribution Summary
Set-aware cross-modal alignment for one-to-many sonar-optical correspondences.
Geometry-aware contrastive supervision from free metadata (overlap priors).
Reproducible unlabeled evaluation stack with multi-seed reporting and paper-ready artifacts.
