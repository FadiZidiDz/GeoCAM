# GeoCAM

**Geometry-weighted cross-modal alignment for unlabeled sonar–optical representation learning.**

GeoCAM learns shared embeddings between side-scan sonar (SSS) tiles and co-registered optical images on BenthiCat SSS-CAM **without class labels or segmentation masks**. The method combines:

- a **Set Attention Aggregator (SAA)** for multi-view optical pooling,
- **geometry-weighted InfoNCE** using geographic overlap priors,
- an optional **cross-modal masked reconstruction (CMMR)** branch (disabled in the final retrieval recipe).

Primary evaluation: cross-modal **retrieval** (R@k, MRR, NDCG@k) on held-out tile pairs.

---

## Repository layout

```
geocam/
  configs/          # YAML training configs (final_paper.yaml, paper_eval.yaml, …)
  data/             # Dataset, overlap precomputation, transforms
  losses/           # Geo-weighted InfoNCE, CMMR reconstruction loss
  matching/         # N07-style matching benchmark (extension)
  models/           # GeoCAM, encoders, SAA, CMMR decoder
  training/         # Pretrain, ablations, paper eval, table/figure scripts
  utils/            # Metrics, logging, checkpoints
  requirements.txt
docs/
  GeoCAM_model_description.md   # Extended model & dataset notes (optional)
```

Run all commands from the **repository root** (the directory that contains `geocam/`).

---

## Requirements

- Python 3.10+
- CUDA GPU recommended for training
- BenthiCat SSS-CAM data (local path; not included in this repo)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r geocam/requirements.txt
```

Set `PYTHONPATH` to the repo root if imports fail:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"   # Windows PowerShell: $env:PYTHONPATH = "$(pwd)"
```

---

## Data preparation

GeoCAM expects a BenthiCat-style layout:

| File / folder | Role |
|---------------|------|
| `sonar/sonar/*.npy` | Sonar tiles |
| `camera/camera/*.png` | Optical images |
| `correspondences.json` | Tile → optical image links |
| `sonar/sonar/sonar.csv`, `camera/camera/camera.csv` | Footprint metadata |
| `overlap_weights.json` | Precomputed overlap ratios (required for training) |

1. Point `data.data_root` in your config to your dataset root (see `geocam/configs/final_paper.yaml`).

2. Precompute overlap weights (once per dataset):

```bash
python -m geocam.data.precompute_overlaps --data_root /path/to/BenthiCat/data
```

---

## Training

### Final recipe (100 epochs)

Edit `geocam/configs/final_paper.yaml` (`data_root`, seeds, etc.), then:

```bash
python -m geocam.training.pretrain \
  --config geocam/configs/final_paper.yaml \
  --output_dir geocam_runs/final_paper/seed_42
```

Key defaults in the final recipe: **SAA on**, **geo-weighted InfoNCE on**, **CMMR off** (`enable_cmmr_masking: false`, `lambda_cmmr: 0`), **temperature τ = 0.15**, **100 epochs**, batch size 64.

### Quick smoke test

```bash
python -m geocam.training.smoke_model
```

---

## Paper evaluation (multi-seed)

Full protocol (seeds 42, 52, 62; 100 epochs) is defined in `geocam/configs/paper_eval.yaml`:

```bash
python -m geocam.training.run_paper_eval --paper_config geocam/configs/paper_eval.yaml
python -m geocam.training.aggregate_paper_results --runs_root geocam_runs/paper_eval
python -m geocam.training.make_paper_figures --runs_root geocam_runs/paper_eval
python -m geocam.training.make_paper_tables --runs_root geocam_runs/paper_eval
```

Each run writes `training_report.json`, `training_report.txt`, and `epoch_metrics.csv` under `geocam_runs/paper_eval/<variant>/seed_*/`.

---

## Block ablations

Short-budget component ablations (seed 42, 10 epochs):

```bash
python -m geocam.training.run_block_ablation_seed42.py
python -m geocam.training.make_block_ablation_table.py
```

Outputs LaTeX/CSV/MD under `geocam_runs/ablation_seed42_e10/`. The last table row can pull full-model metrics from `geocam_runs/paper_eval/final_paper/`.

Variants: MeanPool baseline → SAA only → + geo-weight → + CMMR → + hard negatives → final GeoCAM recipe.

---

## Matching benchmark (optional extension)

```bash
python -m geocam.training.run_matching_bench --config geocam/configs/matching_bench.yaml
python -m geocam.training.package_matching_artifacts
```

Protocol pairs: `geocam/matching/protocols/n07_pairs.csv`.

---

## Metrics

Validation logs include:

- **Retrieval:** `val/R@1`, `val/R@5`, `val/R@10` (mean of sonar→optical and optical→sonar unless noted)
- **Ranking:** `val/s2o/MRR`, `val/s2o/MedianRank`
- **NDCG:** `val/s2o/NDCG@5`, `val/s2o/NDCG@10`
- **Diagnostics:** `val/diag/pos_minus_neg`, `val/diag/saa_entropy`, etc.

Best checkpoint is selected by validation loss.

---

## Citation

If you use this code, please cite the GeoCAM paper (BibTeX to be added).

---

## License

Add a `LICENSE` file before public release (e.g. MIT or your institution’s policy).

---

## Acknowledgements

- BenthiCat SSS-CAM dataset and correspondences
- DINOv2 (optical backbone, frozen during SSL)
- timm ViT-S/16 (sonar encoder)
