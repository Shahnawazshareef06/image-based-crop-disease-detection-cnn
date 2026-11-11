# ViT-Tiny Training and Evaluation — PlantVillage (38 Classes)

## 1. Project Overview

This repository provides an enterprise-grade, fully reproducible implementation for developing, training, and evaluating a **Vision Transformer Tiny (`vit_tiny_patch16_224`)** on the 38-class PlantVillage crop disease classification dataset.

The workflow:
1. Reuses the team's verified dataset splits and GPU augmentation pipeline.
2. Evaluates the top 5 ranked augmentation configurations plus a no-augmentation baseline across three fixed random seeds (`42`, `123`, `2026`).
3. Computes rigorous validation metrics (Accuracy, Macro Precision, Macro Recall, Macro-F1, Weighted F1, Loss).
4. Conducts deep validation error analysis (per-class metrics, confusion matrix, weakest classes, confused pairs).
5. Compiles validation metrics, benchmark reports, and diagnostic error analysis tables.

---

## 2. Model Architecture

- **Backbone Architecture**: `vit_tiny_patch16_224` (Vision Transformer Tiny)
- **Library**: `timm` (PyTorch)
- **Pretrained Weights**: ImageNet-1K (`pretrained=True`)
- **Input Resolution**: 224 × 224 pixels
- **Input Channels**: 3 (RGB)
- **Patch Size**: 16 × 16
- **Embedding Dimension**: 192
- **Transformer Depth**: 12 layers
- **Attention Heads**: 3
- **Number of Classes**: 38 (custom linear classification head replacing the 1000-class head)
- **Trainable Scope**: Full fine-tuning (backbone and classifier head trainable)
- **Input Normalization**:
  - Mean: `(0.485, 0.456, 0.406)`
  - Standard Deviation: `(0.229, 0.224, 0.225)`

---

## 3. Dataset Specification

Authoritative split manifest: `split.csv`

| Split | Number of Images | Percentage | Classes Represented |
|---|---|---|---|
| **Training** | 37,751 | ~70.3% | 38 / 38 |
| **Validation** | 7,976 | ~14.8% | 38 / 38 |
| **Testing** | 7,978 | ~14.9% | 38 / 38 |
| **Total** | 53,705 | 100.0% | 38 / 38 |

### Integrity Verification Highlights
- **Duplicates**: 0 duplicate paths across all 53,705 rows.
- **Data Leakage**: 0 overlapping paths between training, validation, and testing sets.
- **Folder / Label Consistency**: 100% match between path directory names and class labels.
- **Class Mapping**: Exactly matches the project standard `class_to_idx.json` (sorted alphabetically).

---

## 4. Fixed Hyperparameters

| Hyperparameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Backbone Learning Rate** | `1e-5` |
| **Classifier Learning Rate** | `5e-4` |
| **Weight Decay** | `1e-4` |
| **Loss Function** | CrossEntropyLoss |
| **Epochs** | 3 |
| **Training Batch Size** | 32 |
| **Validation Batch Size** | 64 |
| **Random Seeds** | 42, 123, 2026 |
| **Mixed Precision** | FP16 (`torch.amp.autocast('cuda')` with `GradScaler`) |
| **LR Scheduler** | None |
| **DataLoader Workers** | 4 |
| **Prefetch Factor** | 4 |
| **Parameter Groups** | Separate groups for backbone (`1e-5`) and classifier head (`5e-4`) |

---

## 5. Augmentation Configurations (Top 5 + Baseline)

Extracted from `terravista_all_logged_configs_ranked.json`:

| Recipe Name | p_flip | p_rotation | p_colour | p_blur | p_noise | p_cutout | p_background | ResNet-50 Val Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| `00_no_augmentation` (Baseline) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.984665 ± 0.000850 |
| `03_uniform_p010` (Rank 2) | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.984099 ± 0.002207 |
| `02_uniform_p020` (Rank 3) | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.983800 ± 0.001637 |
| `01_joint_refined_recipe` (Rank 4) | 0.00 | 0.00 | 0.25 | 0.10 | 0.25 | 0.25 | 0.10 | 0.983067 ± 0.002403 |
| `04_uniform_p025` (Rank 5) | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.982333 ± 0.002532 |

*Note*: Augmentations are applied exclusively to training samples via GPU acceleration. Validation samples undergo deterministic resize and ImageNet normalization only.

---

## 6. Project Structure

```
├── config.py                                  # Centralized hyperparameter and dataset configuration
├── model.py                                   # ViT-Tiny architecture, parameter groups, benchmark
├── dataset.py                                 # Dataset loading, manifest verification, DataLoaders
├── train.py                                   # Training loop, multi-seed runner, checkpoint saving
├── evaluate.py                                # Metrics computation, classification report, error analysis
├── augmentation_pipeline.py                   # PyTorch GPU augmentation operations
├── sanity_check.py                            # Pipeline & forward pass validation script
├── benchmark_model.py                         # Parameter count and latency profiling script
├── prepare_handover_artifacts.py              # Script generating evaluation metrics and reports
├── requirements.txt                           # Pinned project dependencies
├── README.md                                  # Technical project documentation
├── MODEL_RESULTS_AND_SUMMARY.md               # Complete weekly model results summary
├── split.csv                                  # Authoritative 53,705-image split manifest
├── class_to_idx.json                          # Sorted 38-class mapping
├── vit_tiny_top5_configs.json                 # Extracted top 5 augmentation configurations
├── mean_std_results.csv                       # Augmentation ranking & multi-seed statistics
├── vit_tiny_top5_augmentation_results.csv     # Formatted augmentation comparison table
├── per_seed_results.csv                       # Granular per-seed per-recipe metrics
├── classification_report.csv                  # Full 38-class precision, recall, F1, and support
├── weakest_classes.csv                        # Top 5 lowest-F1 disease classes
├── confused_class_pairs.csv                   # Top 10 confused disease pairs
├── validation_predictions.csv                 # Sample-level validation prediction log
├── vit_tiny_benchmark_report.json             # Hardware latency, throughput, and parameter metrics
└── confusion_matrix.png                       # 300 DPI high-resolution confusion matrix heatmap
```

---

## 7. Reproduction Instructions

### Step 1: Environment Setup
```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
```

### Step 2: Run Full Training (All Recipes × 3 Seeds)
```bash
python train.py \
  --split-csv split.csv \
  --class-to-idx class_to_idx.json \
  --image-root /path/to/PlantVillage/color \
  --output-dir ./vit_tiny_runs
```

### Step 3: Run Baseline Only
```bash
python train.py \
  --recipes 00_no_augmentation \
  --seeds 42 123 2026 \
  --output-dir ./vit_tiny_runs
```

### Step 4: Evaluate a Checkpoint
```bash
python evaluate.py \
  --checkpoint vit_tiny_runs/00_no_augmentation_seed_42/best_vit_tiny_00_no_augmentation_seed_42.pth \
  --split-csv split.csv \
  --class-to-idx class_to_idx.json \
  --image-root /path/to/PlantVillage/color \
  --output-dir ./evaluation_results
```

---

## 8. Test Set Restriction

> [!IMPORTANT]
> The test split (`7,978` images) is strictly reserved for final unbiased benchmark reporting. It must never be evaluated during augmentation selection, model tuning, or hyperparameter validation.
>
> The test evaluation requires an explicit command-line flag:
> ```bash
> python evaluate.py --checkpoint <path> --evaluate-test
> ```
> This flag is not invoked during standard training or validation runs.

---

## 9. Model Evaluation Artifacts

All evaluation artifacts are generated directly in the repository root:
- Multi-seed comparison tables (`mean_std_results.csv`, `per_seed_results.csv`).
- Comprehensive classification report and confusion matrix visualization (`classification_report.csv`, `confusion_matrix.png`).
- Diagnostics identifying weakest performing disease classes and confused pairs (`weakest_classes.csv`, `confused_class_pairs.csv`).
- Detailed model summary in `MODEL_RESULTS_AND_SUMMARY.md`.
