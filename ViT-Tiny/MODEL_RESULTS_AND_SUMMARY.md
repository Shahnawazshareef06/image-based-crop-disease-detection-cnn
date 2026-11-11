# ViT-Tiny Model Results & Work Summary

**Project**: PlantVillage 38-Class Foliar Disease Classification  
**Model**: Vision Transformer Tiny (`vit_tiny_patch16_224`)  
**Study Scope**: Baseline vs. Top 5 GPU Augmentation Recipes Across 3 Seeds (`42`, `123`, `2026`)  
**Scope Notice**: Strictly model architecture, training, evaluation, and benchmark results.  
**Report Date**: November 1, 2026  

---

## 1. Executive Summary & Accomplishments

This week's work centered exclusively on developing, training, validating, and benchmarking the **Vision Transformer Tiny (`vit_tiny_patch16_224`)** architecture on the 38-class PlantVillage dataset.

### Key Milestones Completed:
1. **Model Architecture Implementation**: Built an end-to-end PyTorch ViT-Tiny pipeline via `timm` with differential parameter groups (backbone learning rate `1e-5`, classifier head learning rate `5e-4`).
2. **Dataset Audit & Split Verification**: Verified the authoritative 53,705-image `split.csv` manifest, ensuring zero train-val-test overlap, zero duplicate file paths, and 100% alphabetical class label alignment.
3. **Multi-Seed Augmentation Experimentation**: Conducted multi-seed training and evaluation comparing 5 ranked GPU-accelerated augmentation recipes against an unaugmented baseline across three fixed random seeds (`42`, `123`, `2026`).
4. **Validation Error & Confusion Diagnostics**: Evaluated 7,976 validation images to extract per-class classification metrics, identify the 5 lowest-performing disease classes, catalog the top 10 confused disease pairs, and produce a 300 DPI 38×38 confusion matrix.
5. **Inference Latency & Hardware Profiling**: Benchmarked single-image latency (97.94 ms on CPU), batched inference throughput (31.2 images/s), parameter counts (5.53M), and checkpoint storage (21.16 MB).
6. **Git Release Organization**: Organized all model code, configuration JSONs, and evaluation CSV artifacts for clean repository commits.

---

## 2. Model Architecture & PyTorch Configuration

- **Backbone Architecture**: `vit_tiny_patch16_224` (via `timm`)
- **Pretrained Weights**: ImageNet-1K (`pretrained=True`)
- **Input Image Size**: 224 × 224 pixels, RGB (3 channels)
- **Patch Resolution**: 16 × 16 (196 visual tokens + 1 class token)
- **Embedding Dimension**: 192
- **Transformer Encoder Depth**: 12 Transformer blocks
- **Self-Attention Heads**: 3 heads per block
- **Classification Head**: `Linear(in_features=192, out_features=38)`
- **Parameter Breakdown**:
  - Backbone parameters: 5,524,416
  - Classification head parameters: 7,334
  - **Total parameters**: 5,531,750 (~5.53 M)
  - **Trainable parameters**: 5,531,750 (100% full fine-tuning)
- **Checkpoint Footprint**: ~21.16 MB on disk
- **Input Normalization**:
  - Mean: `(0.485, 0.456, 0.406)`
  - Standard Deviation: `(0.229, 0.224, 0.225)`

---

## 3. Dataset Specification & Split Manifest

All experiments were executed strictly on the verified `split.csv` manifest:

| Split | Number of Images | Percentage | Classes Represented | Purpose |
|---|---|---|---|---|
| **Training** | 37,751 | 70.30% | 38 / 38 | Model training with GPU augmentations |
| **Validation** | 7,976 | 14.85% | 38 / 38 | Model evaluation, recipe selection, error analysis |
| **Testing** | 7,978 | 14.85% | 38 / 38 | **Held-out split**; reserved strictly for final benchmark |
| **Total** | **53,705** | **100.0%** | **38 / 38** | Zero duplicates; zero inter-split data leakage |

- **Class Mapping**: 38 classes alphabetically indexed from `Apple___Apple_scab` (`0`) to `Tomato___Tomato_mosaic_virus` (`37`), stored in `class_to_idx.json`.

---

## 4. Hyperparameter Specifications

| Hyperparameter | Value | Implementation Details |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay optimizer |
| **Backbone LR** | `1e-5` | Preserves ImageNet transfer representations |
| **Classifier Head LR** | `5e-4` | Accelerated convergence for the 38-class linear head |
| **Weight Decay** | `1e-4` | Applied across all parameter groups |
| **Loss Function** | CrossEntropyLoss | Plain multi-class cross-entropy |
| **Epochs** | 3 | Standard benchmark protocol |
| **Train Batch Size** | 32 | GPU memory-optimized batch |
| **Val Batch Size** | 64 | Fast evaluation throughput |
| **Random Seeds** | `42`, `123`, `2026` | Ensures reproducible evaluation |
| **Precision** | Mixed Precision FP16 | `torch.amp.autocast('cuda')` with `GradScaler` |
| **DataLoader Workers** | 4 | Multi-process image loading |

---

## 5. Augmentation Study (Top 5 Recipes + Baseline)

The 5 evaluated GPU tensor augmentation recipes from `vit_tiny_top5_configs.json` and the unaugmented baseline:

| Recipe Name | Flip | Rotation | Colour Jitter | Gaussian Blur | Gaussian Noise | Cutout | Random BG |
|---|---|---|---|---|---|---|---|
| `00_no_augmentation` (Baseline) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `03_uniform_p010` | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| `02_uniform_p020` | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| `01_joint_refined_recipe` | 0.00 | 0.00 | 0.25 | 0.10 | 0.25 | 0.25 | 0.10 |
| `04_uniform_p025` | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |

*Note: Augmentation is applied exclusively during training on GPU tensors. Validation images undergo deterministic bilinear resizing and ImageNet normalization.*

---

## 6. Model Results & Multi-Seed Comparison

### Quantitative Validation Results Across Seeds 42, 123, 2026
| Rank | Recipe Name | Seed 42 Macro-F1 | Seed 123 Macro-F1 | Seed 2026 Macro-F1 | Mean Macro-F1 | Std Macro-F1 | Mean Accuracy |
|---|---|---|---|---|---|---|---|
| **1** | `00_no_augmentation` | **0.9853** | 0.9850 | 0.9837 | **0.984665** | **±0.000850** | **98.77%** |
| **2** | `03_uniform_p010` | 0.9820 | 0.9839 | **0.9864** | **0.984099** | ±0.002207 | 98.71% |
| **3** | `02_uniform_p020` | 0.9820 | 0.9842 | 0.9852 | **0.983800** | ±0.001637 | 98.68% |
| **4** | `01_joint_refined_recipe` | 0.9832 | **0.9854** | 0.9806 | **0.983067** | ±0.002403 | 98.62% |
| **5** | `04_uniform_p025` | 0.9814 | 0.9804 | 0.9852 | **0.982333** | ±0.002532 | 98.55% |

### Key Experimental Insights:
- **Baseline Stability**: `00_no_augmentation` achieved the highest mean validation Macro-F1 (**0.9847**) and the lowest standard deviation across seeds (**±0.00085**), with 98.77% mean accuracy.
- **Top Augmented Configuration**: `03_uniform_p010` (mild 10% uniform rate) was the strongest augmented recipe (**0.9841** mean Macro-F1), peaking at **0.9864** on seed 2026.
- **Sensitivity to High Distortions**: Higher augmentation rates (`04_uniform_p025`) reduced Macro-F1 to **0.9823**, indicating that excessive blur, noise, and cutout obscure microscopic foliar symptoms.

---

## 7. Model Error Diagnostics & Weakest Classes

Detailed validation predictions on 7,976 images yielded the following per-class insights:

### Top 5 Lowest-Performing Classes
| Class ID | Disease Category | Precision | Recall | F1-Score | Validation Support |
|---|---|---|---|---|---|
| **7** | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 81.33% | 79.22% | **0.8026** | 77 |
| **29** | `Tomato___Early_blight` | 85.16% | 88.00% | **0.8656** | 150 |
| **9** | `Corn_(maize)___Northern_Leaf_Blight` | 89.26% | 90.48% | **0.8986** | 147 |
| **2** | `Apple___Cedar_apple_rust` | 82.00% | 100.00% | **0.9011** | 41 |
| **34** | `Tomato___Target_Spot` | 91.59% | 92.89% | **0.9224** | 211 |

### Top 10 Most Frequently Confused Disease Pairs
| Rank | Actual Class | Predicted Class | Misclassified Instances |
|---|---|---|---|
| 1 | `Tomato___Early_blight` | `Tomato___Target_Spot` | 18 |
| 2 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn_(maize)___Northern_Leaf_Blight` | 16 |
| 3 | `Tomato___Target_Spot` | `Tomato___Early_blight` | 15 |
| 4 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 14 |
| 5 | `Pepper,_bell___Bacterial_spot` | `Tomato___Bacterial_spot` | 12 |
| 6 | `Tomato___Spider_mites Two-spotted_spider_mite` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 11 |
| 7 | `Tomato___Bacterial_spot` | `Pepper,_bell___Bacterial_spot` | 10 |
| 8 | `Apple___Apple_scab` | `Apple___Cedar_apple_rust` | 9 |
| 9 | `Grape___Black_rot` | `Grape___Esca_(Black_Measles)` | 8 |
| 10 | `Potato___Early_blight` | `Tomato___Early_blight` | 8 |

---

## 8. Inference Latency & Hardware Profile

Benchmarked on CPU (reference environment) across 10 warmup runs and 30 test iterations:
- **Total Parameters**: 5,531,750
- **Trainable Parameters**: 5,531,750 (100%)
- **Checkpoint Size**: 21.16 MB
- **Single-Image Latency (Batch=1)**: 97.94 ms
- **Batched Latency (Batch=32)**: 1025.65 ms
- **Inference Throughput**: 31.20 images / second

---

## 9. Model-Only Git Commit Manifest

The table below lists every file to commit for this week's milestone, restricted strictly to **model code, configuration, documentation, and model results**:

### Files to Commit to Git
| Category | File Path | Description |
|---|---|---|
| **Model Source Code** | `model.py` | ViT-Tiny architecture builder, parameter groups, and benchmark function |
| | `config.py` | Model hyperparameters, training settings, and dataset definitions |
| | `dataset.py` | Dataset verification, PyTorch Dataset, and DataLoaders |
| | `train.py` | Model training script with multi-seed execution and FP16 precision |
| | `evaluate.py` | Validation metrics calculator, confusion matrix generator, and error analyzer |
| | `augmentation_pipeline.py` | GPU tensor-level augmentation operations |
| | `sanity_check.py` | Model forward-pass and tensor dimension verification |
| | `benchmark_model.py` | Parameter count, checkpoint size, latency, and throughput profiler |
| | `prepare_handover_artifacts.py` | Script generating model results and evaluation CSVs |
| **Model Configs & Mappings** | `class_to_idx.json` | 38-class alphabetically ordered class-to-index mapping |
| | `vit_tiny_top5_configs.json` | Top 5 augmentation probability configurations |
| | `requirements.txt` | Pinned project dependencies (`torch`, `timm`, `torchvision`, etc.) |
| **Model Results & Reports** | `mean_std_results.csv` | Mean and std Macro-F1 across 3 seeds for all recipes |
| | `vit_tiny_top5_augmentation_results.csv` | Ranked augmentation comparison table |
| | `per_seed_results.csv` | Granular per-seed per-recipe metrics (Accuracy, F1, Loss) |
| | `classification_report.csv` | 38-class precision, recall, F1-score, and support counts |
| | `weakest_classes.csv` | Top 5 lowest-F1 disease classes |
| | `confused_class_pairs.csv` | Top 10 confused disease pairs |
| | `vit_tiny_benchmark_report.json` | Latency, throughput, and parameter metrics |
| | `confusion_matrix.png` | High-resolution 300 DPI 38×38 confusion matrix heatmap |
| **Documentation & Git** | `README.md` | Primary technical documentation and reproduction instructions |
| | `MODEL_RESULTS_AND_SUMMARY.md` | This model-only weekly progress and results report |
| | `.gitignore` | Configured to ignore cache, checkpoints (`*.pth`), and non-model files |

### Explicitly Excluded (Do Not Commit)
- **Checkpoints (`*.pth`)**: `*.pth` binary weights (~21.2 MB each, ~63.6 MB total) to avoid bloating Git history.
- **Python Cache**: `__pycache__/` and `*.pyc` files.
- **Archives**: `TerraVista_Reproducible_final.zip`.

---

## 10. Ready-to-Run Git Commands (Model-Only)

Execute these commands in your project root to commit only the model-related files:

```bash
# 1. Initialize git (if not already initialized)
git init

# 2. Add documentation, configs, and dependencies
git add .gitignore README.md MODEL_RESULTS_AND_SUMMARY.md requirements.txt
git add class_to_idx.json vit_tiny_top5_configs.json

# 3. Add core model code and scripts
git add model.py config.py dataset.py train.py evaluate.py
git add augmentation_pipeline.py sanity_check.py benchmark_model.py prepare_handover_artifacts.py

# 4. Add model results, metric CSVs, and visualizations
git add mean_std_results.csv vit_tiny_top5_augmentation_results.csv per_seed_results.csv
git add classification_report.csv weakest_classes.csv confused_class_pairs.csv
git add vit_tiny_benchmark_report.json confusion_matrix.png

# 5. Commit with an enterprise commit message
git commit -m "feat(vit-tiny): complete ViT-Tiny model architecture, multi-seed training, augmentation benchmark, and error analysis"
```
