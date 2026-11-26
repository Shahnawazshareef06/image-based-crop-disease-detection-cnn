# Focal Loss vs Cross-Entropy: MobileNetV3-Large (frozen, PlantVillage)

Part of the IIIT Hyderabad capstone on image-based crop disease classification. This folder covers the MobileNetV3-Large part of one question: **does focal loss help with class imbalance when only the last layer of a pretrained model is trained?** ResNet-50 and ViT-Tiny are covered in the main README.

Validation data only. The test set was never loaded, resolved or evaluated.

## Short answer

No clear benefit. Focal loss and cross-entropy are statistically indistinguishable on MobileNetV3-Large, and focal loss did not help the five smallest classes.

| Loss | Validation macro-F1 (mean ± std, 3 seeds) | Validation accuracy (mean) | Mean F1, five smallest classes |
|---|---|---|---|
| Cross-entropy | 0.9678 ± 0.0021 | 0.9738 | 0.9710 |
| Focal, gamma 1 | 0.9685 ± 0.0015 | 0.9748 | 0.9697 |
| Focal, gamma 2 | 0.9679 ± 0.0016 | 0.9744 | 0.9669 |

How to read each output file is explained in `RESULTS_EXPLAINED.md` (same file formats as the other two models).

## Setup (identical to the ResNet-50 and ViT-Tiny runs)

- **Data:** PlantVillage color images, 38 classes. Fixed split from `train_val_split.csv`: 37,751 train, 7,976 validation. Do not change the split file or `class_to_idx.json`.
- **Model:** MobileNetV3-Large (torchvision, ImageNet-pretrained weights). The final layer `classifier[3]` is replaced with a 38-class linear layer.
- **Frozen backbone:** only `classifier[3]` is trained: 48,678 of 4,250,710 parameters. Every other layer is kept in evaluation mode, so the dropout layer in the classifier is off during training.
- **Augmentation:** `uniform_p010` recipe, applied on the GPU during training only.
- **Training:** 10 epochs, seeds 42 / 123 / 2026, AdamW, mixed precision.
- **Losses:** cross-entropy, focal gamma 1, focal gamma 2. No class weights (alpha), so focal loss only down-weights easy examples: `loss = -(1 - p_t)^gamma * log(p_t)`.
- **Model selection:** for each run, the epoch with the best validation macro-F1 is reported.

## Repository contents

| Path | What it is |
|---|---|
| `mobilenetv3-notebook.ipynb` | Kaggle notebook that ran everything (Save & Run All, GPU T4, Internet ON) |
| `run_focal_loss_mobilenetv3.py` | Training script from this notebook (adds `--model mobilenetv3`; save under this name so it does not overwrite the ResNet-50 / ViT-Tiny script) |
| `focal_runs_mobilenetv3_frozen_ep10/` | Per-run folders (`<loss>_seed<N>/` with `history.csv`, `per_class_f1.csv`, `run_summary.json`), `run_config.json`, `per_seed_results.csv`, `comparison_table.csv`, `all_38_classes_comparison.csv`, `conclusion_draft.txt` |
| `requirements-lock.txt`, `environment_report.json`, `torch_timm_versions.txt` | Exact package versions |

Notes on the two main result files:
- `per_seed_results.csv`: one row per run (9 rows) with best epoch, macro-F1, accuracy and F1 for all 38 classes.
- `all_38_classes_comparison.csv`: per class, the mean F1 and the standard deviation across seeds for each loss.

## How to reproduce

1. Open a Kaggle notebook with GPU T4 and Internet ON.
2. Add the PlantVillage dataset (the `color` folder) and the team input files (`focal-inputs`).
3. Run the notebook top to bottom. The key command is:

```bash
python run_focal_loss.py --color-dir <path>/plantvillage-dataset/color \
  --model mobilenetv3 --frozen --include-ce --epochs 10 \
  --cache-dir /kaggle/temp/terravista_cache \
  --output-dir ./focal_runs_mobilenetv3_frozen_ep10
```

4. In the log, confirm `mobilenetv3 | trainable parameters: 48,678 of 4,250,710`. A different number means the freezing is wrong.

Notes:
- The script needs a CUDA GPU and stops otherwise.
- It checks the split file and class mapping against `reproducibility_manifest.json` and refuses to run if they differ.
- Finished runs are skipped when the script is run again.
- Runs are seeded, but GPU training is not guaranteed to be bit-for-bit identical across hardware or library versions. This run used torch 2.11.0+cu130; the ResNet-50 and ViT-Tiny runs used torch 2.11.0+cu128.

## Findings

1. **Overall:** all three losses are within 0.0007 macro-F1 of each other, smaller than the seed-to-seed spread (0.0015 to 0.0021). Seed by seed, gamma 1 was higher in two seeds and lower in one; gamma 2 was higher in two and lower in one.
2. **Smallest classes:** the five smallest training classes are Potato healthy (105 images), Apple Cedar apple rust (192), Raspberry healthy (248), Peach healthy (251) and Tomato mosaic virus (257). Focal loss lowered their mean F1 slightly (0.9710 to 0.9697 and 0.9669). Potato healthy was unchanged at gamma 1 (0.9285) and fell at gamma 2 (0.9208). Tomato mosaic virus fell from 0.9531 to 0.9462 and 0.9401, and was the biggest loss for both gammas.
3. **Per class:** focal loss beat cross-entropy on 21 classes at gamma 1 and 20 at gamma 2 (6 and 10 worse, the rest tied), but the average change is +0.0007 and +0.0001. These are many tiny wins and a few small losses. The largest gain is +0.0067 (Pepper bell healthy); the largest loss is -0.0129 (Tomato mosaic virus, gamma 2).
4. **Hardest classes** under every loss: Tomato Early blight (about 0.85), Corn Cercospora/Gray leaf spot (about 0.86), Tomato Target spot (about 0.87). These look like confusable diseases, not rare classes, so focal loss does not fix them.
5. **Across models (cross-entropy):** ResNet-50 0.9708, MobileNetV3-Large 0.9678, ViT-Tiny 0.9555. The gap between models is much larger than any loss effect.

## Limitations

- **Only 3 seeds**, so small differences cannot be tested reliably.
- **Not converged:** the best epoch is 10 in all 9 runs. Validation scores were still rising at the end, so more epochs could change the picture.
- **Head-only training:** just 48,678 trainable parameters. These results say nothing about focal loss with full fine-tuning.
- **Validation set only**, and the best epoch is picked on it, so the numbers are slightly optimistic. No test numbers were used.
- **Tiny classes are noisy:** Potato healthy has only about 20 validation images (estimated from the split ratio), so one image moves its F1 by roughly 0.04.
- **No class weighting** (alpha) was tried with focal loss.
