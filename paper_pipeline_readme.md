# GeoCAM Unlabeled SSL Paper Pipeline

This pipeline automates full-budget (100 epoch) multi-seed retrieval evaluation for the unlabeled BenthiCat setting and exports publication-ready artifacts.

## Protocol

- Seeds: `42, 52, 62`
- Full budget: `100` epochs per seed
- Variants: configured in `geocam/configs/paper_eval.yaml`
- Outputs include run-level reports, aggregated summary tables, and figures

## Step 1: Run full protocol

```bash
python -m geocam.training.run_paper_eval --paper_config geocam/configs/paper_eval.yaml
```

Optional dry run (quick wiring check):

```bash
python -m geocam.training.run_paper_eval --paper_config geocam/configs/paper_eval.yaml --dry_run --max_train_batches 1 --max_val_batches 1
```

## Step 2: Aggregate seed results

```bash
python -m geocam.training.aggregate_paper_results --runs_root geocam_runs/paper_eval
```

Generated:
- `geocam_runs/paper_eval/paper_results_summary.csv`
- `geocam_runs/paper_eval/paper_results_summary.md`

## Step 3: Generate figures

```bash
python -m geocam.training.make_paper_figures --runs_root geocam_runs/paper_eval
```

Generated:
- `figures/retrieval_curves.png` (`R@1/R@5/R@10`)
- `figures/val_infonce_curve.png`
- `figures/pos_vs_neg_similarity.png`
- `figures/saa_entropy_curve.png`

## Step 4: Build core ablation tables

```bash
python -m geocam.training.make_paper_tables --runs_root geocam_runs/paper_eval
```

Generated:
- `core_ablation_table.csv`
- `core_ablation_table.md`

## Exported metrics

The pretrain evaluation exports:

- Retrieval: `R@1`, `R@5`, `R@10`
- Ranking: `MRR`, `MedianRank`
- Graded relevance: `NDCG@k` (including `NDCG@5` and `NDCG@10`)
- Diagnostics: `pos_sim`, `neg_sim`, `pos_minus_neg`, `alignment`, `mean_weight`, `saa_entropy`

Each run writes:
- `training_report.json`
- `training_report.txt`
- `epoch_metrics.csv`

## Claim guardrails for unlabeled-only paper

Safe claims:

- "Improves cross-modal retrieval in the unlabeled SSS-optical setting."
- "Improves geometric alignment quality between sonar and optical embeddings."
- "Learns transferable unlabeled representations suitable for downstream adaptation."

Claims to avoid without labeled downstream benchmarks:

- Any segmentation IoU or class-level segmentation improvement claims.
- Habitat map quality improvement claims tied to labels.
- Claims of state-of-the-art downstream classification/segmentation performance.
