# Focal Loss vs Cross-Entropy for Class Imbalance (PlantVillage)

Part of the IIIT Hyderabad capstone on image-based crop disease classification. This folder covers one question: **does focal loss help with class imbalance when only the last layer of a pretrained model is trained?**

Validation data only. The test set was never loaded, resolved or evaluated.

## Short answer

No clear benefit. Focal loss is statistically indistinguishable from cross-entropy on ResNet-50 and slightly worse on ViT-Tiny. It also did not help the five smallest classes.

| Model | Loss | Validation macro-F1 (mean ± std, 3 seeds) | Validation accuracy (mean) |
|---|---|---|---|
| ResNet-50 | Cross-entropy | 0.9708 ± 0.0004 | 0.9761 |
| ResNet-50 | Focal, gamma 1 | 0.9712 ± 0.0007 | 0.9770 |
| ResNet-50 | Focal, gamma 2 | 0.9709 ± 0.0010 | 0.9769 |
| ViT-Tiny | Cross-entropy | 0.9555 ± 0.0009 | 0.9604 |
| ViT-Tiny | Focal, gamma 1 | 0.9541 ± 0.0014 | 0.9587 |
| ViT-Tiny | Focal, gamma 2 | 0.9519 ± 0.0014 | 0.9567 |

Full explanation of every output file is in [`RESULTS_EXPLAINED.md`](RESULTS_EXPLAINED.md).

## Setup (identical for every run)

- **Data:** PlantVillage color images, 38 classes. Fixed split from `train_val_split.csv`: 37,751 train, 7,976 validation. Do not change the split file or `class_to_idx.json`.
- **Models:** ResNet-50 (torchvision weights) and ViT-Tiny (`vit_tiny_patch16_224` from timm), both ImageNet-pretrained.
- **Frozen backbone:** only the final layer is trained. ResNet-50: 77,862 of 23,585,894 parameters. ViT-Tiny: 7,334 of 5,531,750 parameters. Frozen layers are kept in evaluation mode.
- **Augmentation:** `uniform_p010` recipe, applied on the GPU during training only.
- **Training:** 10 epochs, seeds 42 / 123 / 2026, AdamW, mixed precision.
- **Losses:** cross-entropy, focal gamma 1, focal gamma 2. No class weights (alpha) are used in the focal loss, so it only down-weights easy examples.
- **Model selection:** for each run, the epoch with the best validation macro-F1 is reported (`best_epoch`).

Focal loss as implemented: `loss = -(1 - p_t)^gamma * log(p_t)`, averaged over the batch, where `p_t` is the predicted probability of the true class. Gamma 0 is plain cross-entropy.

## Repository contents

| Path | What it is |
|---|---|
| `focal-loss-resnet50-vity-tiny.ipynb` | Kaggle notebook that ran everything (Save & Run All, GPU T4, Internet ON) |
| `run_focal_loss.py` | Training and evaluation script (the version that produced these results) |
| `focal_runs_frozen_ep10/` | ResNet-50 results: per-run folders, `comparison_table.csv`, `per_seed_results.csv`, `conclusion_draft.txt`, `run_config.json` |
| `focal_runs_vit_tiny_frozen_ep10/` | Same structure for ViT-Tiny |
| `all_models_comparison.csv` | Both models' comparison tables merged |
| `all_models_per_seed_results.csv` | Both models' per-seed rows merged |
| `all_38_classes_per_seed.csv` | Per-class F1 for every model, loss and seed |
| `all_38_classes_mean.csv` | Per-class F1 averaged over seeds, plus focal-minus-cross-entropy differences |
| `all_38_classes_std.csv` | Per-class F1 standard deviation across seeds (export from the workbook sheet) |
| `requirements-lock.txt`, `environment_report.json`, `torch_timm_versions.txt` | Exact package versions |

Input files bundled for reproducibility (copied from the team handoff): `train_val_split.csv`, `class_to_idx.json`, `reproducibility_manifest.json`, `augmentation_pipeline.py`, `terravista_*.py`, `verify_handoff.py`, `freeze_environment.py`.

## How to reproduce

1. Open a Kaggle notebook with GPU T4 and Internet ON.
2. Add the PlantVillage dataset (the `color` folder) and the team input files (`focal-inputs`).
3. Run the notebook top to bottom. The two key commands are:

```bash
python run_focal_loss.py --color-dir <path>/plantvillage-dataset/color \
  --model resnet50 --frozen --include-ce --epochs 10 \
  --cache-dir /kaggle/temp/terravista_cache --output-dir ./focal_runs_frozen_ep10

python run_focal_loss.py --color-dir <path>/plantvillage-dataset/color \
  --model vit_tiny --frozen --include-ce --epochs 10 \
  --cache-dir /kaggle/temp/terravista_cache --output-dir ./focal_runs_vit_tiny_frozen_ep10
```

4. The last notebook cells build the merged CSVs and the 38-class tables.

Notes:
- The script needs a CUDA GPU and stops otherwise.
- It checks the split file and class mapping against `reproducibility_manifest.json` and refuses to run if they differ.
- Finished runs are skipped when the script is run again (it checks `run_summary.json`).
- Runs are seeded but GPU training is not guaranteed to be bit-for-bit identical across hardware, so small differences are possible.

## Findings

1. **ResNet-50:** all three losses land within 0.0004 of each other, which is inside the seed-to-seed spread. No real difference.
2. **ViT-Tiny:** focal loss is lower, and the drop grows with gamma (-0.0014 for gamma 1, -0.0036 for gamma 2). Gamma 2 is clearly outside the spread; gamma 1 is borderline.
3. **Smallest classes:** the five smallest training classes are Potato healthy (105 images), Apple Cedar apple rust (192), Raspberry healthy (248), Peach healthy (251) and Tomato mosaic virus (257). Focal loss did not improve their mean F1 for ResNet-50 (0.9749 cross-entropy vs 0.9701 and 0.9677) and was about equal for ViT-Tiny (0.9716 vs 0.9728 and 0.9714).
4. **Per class:** gains and losses are mostly under 0.01 F1 and scatter in both directions. The hardest classes (Corn Cercospora/Gray leaf spot, Tomato Early blight, Tomato Target spot) stay the hardest under every loss.

## Limitations

- **Only 3 seeds**, so small differences cannot be tested reliably.
- **Not converged:** the best epoch is 8, 9 or 10 in every run, and most are 9 or 10. Validation scores were still rising at epoch 10, so more epochs could change the picture.
- **Head-only training** is a very small search space (7,334 to 77,862 trainable parameters). These results do not say anything about focal loss with full fine-tuning.
- **Validation set is the only evaluation set**, and the best epoch is picked on it, so the numbers are slightly optimistic. No test numbers were used.
- **Tiny classes are noisy:** Potato healthy has only about 20 validation images (estimated from the split ratio), so one image changes its F1 by roughly 0.04.
- **No class weighting** (alpha) was tried with focal loss.
- **MobileNetV3-Large** is not part of these results. It has not been run.
