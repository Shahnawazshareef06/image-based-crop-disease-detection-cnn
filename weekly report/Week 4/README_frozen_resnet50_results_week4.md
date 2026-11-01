# TerraVista: Frozen ResNet-50 Augmentation Results (p010 vs p020)

Frozen-backbone ResNet-50 on PlantVillage (38 classes), trained with the TerraVista
augmentation pipeline (`augmentation_pipeline.py`, unmodified; SHA-256 `82a7fe1b…8224a`).
Two augmentation recipes were run with identical settings: `uniform_p010` and `uniform_p020`.

## 1. Headline results

Metric: best validation macro-F1 over 3 epochs, per seed. Mean and standard deviation are
over seeds 42, 123 and 2026. Std is the **sample standard deviation** (n = 3, ddof = 1).

| Recipe | Mean macro-F1 | Std | Min – Max | Mean val accuracy (best epoch) |
|---|---:|---:|---|---:|
| `uniform_p010` | **0.944699** | 0.002060 | 0.942412 – 0.946408 | 0.954530 |
| `uniform_p020` | 0.939202 | 0.001745 | 0.937187 – 0.940260 | 0.949557 |
| **Difference (p010 − p020)** | **+0.005497** | 0.000659 (paired) | +0.005018 – +0.006249 | +0.004973 |

`uniform_p010` scored higher than `uniform_p020` in all three seeds.

## 2. Per-seed results

| Seed | p010 macro-F1 | p020 macro-F1 | Difference | p010 accuracy | p020 accuracy |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.945278 | 0.940260 | +0.005018 | 0.954363 | 0.950100 |
| 123 | 0.942412 | 0.937187 | +0.005225 | 0.953360 | 0.948847 |
| 2026 | 0.946408 | 0.940158 | +0.006249 | 0.955868 | 0.949724 |
| **Mean** | **0.944699** | **0.939202** | **+0.005497** | 0.954530 | 0.949557 |
| **Std** | 0.002060 | 0.001745 | 0.000659 | 0.001262 | 0.000643 |

The best epoch was epoch 3 for every seed in both recipes.

## 3. Learning curves (mean over 3 seeds)

| Epoch | p010 val macro-F1 | p020 val macro-F1 | Gap (p010 − p020) |
|---:|---:|---:|---:|
| 1 | 0.877466 | 0.863981 | +0.013486 |
| 2 | 0.929138 | 0.922339 | +0.006799 |
| 3 | 0.944699 | 0.939202 | +0.005498 |

| Epoch-3 metric (mean over seeds) | p010 | p020 |
|---|---:|---:|
| Train loss | 0.2644 | 0.3151 |
| Train accuracy | 0.9431 | 0.9297 |
| Validation loss | 0.2016 | 0.2187 |

## 4. Comparison and interpretation

- **p010 beat p020 by about 0.0055 macro-F1** (about 0.58% relative). The gap has the same
  sign in all three seeds and is roughly 2.7 times the seed-to-seed std of either recipe
  (about 0.002), so it is unlikely to be seed noise. With only 3 seeds, treat this as
  strong screening evidence, not a formal significance result.
- **The gap shrinks each epoch** (0.0135, then 0.0068, then 0.0055), and both recipes were
  still improving at epoch 3. p020 may narrow the gap, or even catch up, with more epochs.
  This run cannot tell.
- **Heavier augmentation slowed fitting.** p020 had higher training loss and lower
  training accuracy than p010 at every epoch, consistent with a frozen linear head
  having limited capacity to absorb stronger augmentation in 3 epochs.
- **The ordering matches the original full fine-tuning study** (p010 above p020), but the
  gap is larger here.

## 5. Experimental setup (identical for both recipes)

| Item | Value |
|---|---|
| Dataset | PlantVillage `color`, 38 classes, resized to 224 × 224 |
| Split | bundled `train_val_split.csv` (SHA-256 `e3e2d22d…7575a`) |
| Rows | 37,751 train / 7,976 validation |
| Test data | **not used** (non-train/val rows dropped before any path is resolved) |
| Model | torchvision ResNet-50, ImageNet-pretrained, new `Linear(2048, 38)` head |
| Frozen backbone | all backbone weights `requires_grad=False`; network kept in `eval()` mode (BatchNorm statistics fixed) |
| Trainable parameters | 77,862 of 23,585,894 |
| Optimizer | AdamW, lr 5e-4, weight decay 1e-4 (head only) |
| Loss | CrossEntropyLoss (no class weights, no focal loss) |
| Scheduler | none |
| Batch size | 32 train / 64 validation |
| Epochs | 3 |
| Seeds | 42, 123, 2026 |
| Precision | CUDA fp16 autocast + GradScaler |
| Hardware / software | Tesla T4, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 |

### Augmentation recipes

Both recipes apply all seven families, each sampled independently per image on every
training batch. Validation images receive no random augmentation.

| Recipe | flip | rotation | colour | blur | noise | cutout | background |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform_p010` | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| `uniform_p020` | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |

Fixed strengths: rotation ±15°, brightness 0.60–1.40, contrast 0.80–1.20, saturation
0.80–1.20, hue ±0.02, noise std 0.02, 5 × 5 blur, cutout 2–15% of the area, GrabCut
background masks at 112 × 112.

## 6. Reproducing

```bash
python run_frozen_resnet50.py --color-dir /path/to/PlantVillage/color \
  --recipes uniform_p010 --output-dir ./frozen_uniform_p010

python run_frozen_resnet50.py --color-dir /path/to/PlantVillage/color \
  --recipes uniform_p020 --output-dir ./frozen_uniform_p020
```

Each run writes `seed_results.csv`, `frozen_summary.json`, and per-seed `history.csv`,
`run_summary.json` and `best_fc_checkpoint.pt` (head weights only).

## 7. Caveats

- Only 2 of the 5 handoff recipes have been run. `no_augmentation`, `joint_refined` and
  `uniform_p025` are still to do, so these results do not show whether augmentation helps
  the frozen model at all.
- These scores are **not comparable** to the full fine-tuning study (about 0.984), which
  updated all ResNet-50 layers.
- The best epoch is selected on the same validation set that reports the score. Because
  the best epoch was always the last one, the effect here is negligible.
- Three epochs is a short budget for a frozen head; a longer equal-budget run (for
  example 10 epochs) would be needed before treating the ranking as converged.
- Bit-for-bit reproducibility is not guaranteed across different GPUs or library versions.
