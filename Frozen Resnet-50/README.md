# Frozen ResNet-50: Augmentation Recipe Comparison

Results from a controlled comparison of data-augmentation recipes for an image classifier built on a **frozen ResNet-50 backbone** (kept in eval mode) with only the final `fc` layer trained.

Each recipe was trained for 3 epochs on 3 random seeds (42, 123, 2026) and scored by **validation macro-F1**. The test set was **not used** in any of these runs.

## Results summary

Validation macro-F1 at each run's best epoch (n = 3 seeds per recipe). Std and variance are **sample** statistics (ddof = 1).

| Recipe | Mean | Std | Variance | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---|---|---|---|---|
| `no_augmentation` | **0.9500** | 0.0009 | 7.51e-07 | 0.9497 | 0.9494 | 0.9510 |
| `joint_refined` | 0.9456 | 0.0030 | 8.74e-06 | 0.9449 | 0.9431 | 0.9488 |
| `uniform_p010` | 0.9447 | 0.0021 | 4.24e-06 | 0.9453 | 0.9424 | 0.9464 |
| `uniform_p020` | 0.9392 | 0.0017 | 3.05e-06 | 0.9403 | 0.9372 | 0.9402 |
| `uniform_p025` | 0.9370 | 0.0026 | 6.55e-06 | 0.9380 | 0.9341 | 0.9389 |

![Mean and std per recipe](figures/01_mean_std_bar.png)

*Note: the y-axis starts at 0.925 to make differences visible, so bar heights exaggerate the gaps. The real spread between best and worst recipe is about 1.3 F1 points.*

## Detailed descriptive statistics (validation macro-F1)

| Recipe | Mean | Median | Std | Variance | Min | Max | Range | CV (%) | SEM | 95% CI (t, df = 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| `no_augmentation` | 0.9500 | 0.9497 | 0.0009 | 7.51e-07 | 0.9494 | 0.9510 | 0.0016 | 0.091 | 0.0005 | [0.9479, 0.9522] |
| `joint_refined` | 0.9456 | 0.9449 | 0.0030 | 8.74e-06 | 0.9431 | 0.9488 | 0.0058 | 0.313 | 0.0017 | [0.9383, 0.9529] |
| `uniform_p010` | 0.9447 | 0.9453 | 0.0021 | 4.24e-06 | 0.9424 | 0.9464 | 0.0040 | 0.218 | 0.0012 | [0.9396, 0.9498] |
| `uniform_p020` | 0.9392 | 0.9402 | 0.0017 | 3.05e-06 | 0.9372 | 0.9403 | 0.0031 | 0.186 | 0.0010 | [0.9349, 0.9435] |
| `uniform_p025` | 0.9370 | 0.9380 | 0.0026 | 6.55e-06 | 0.9341 | 0.9389 | 0.0048 | 0.273 | 0.0015 | [0.9307, 0.9434] |

CV = std / mean x 100. SEM = std / sqrt(n). With only 3 seeds the t-intervals are wide, so treat them as rough.

![95% confidence intervals](figures/02_confidence_intervals.png)

![Variability across seeds](figures/04_variability.png)

`no_augmentation` is the most stable recipe (lowest std, variance and CV). `joint_refined` has the highest seed-to-seed variability, driven largely by its seed-2026 score (0.9488) sitting well above seeds 42 and 123.

## Other validation metrics (mean ± std over seeds, at best epoch)

| Recipe | Accuracy | Macro precision | Macro recall | Weighted F1 | Val loss | Train loss | Train acc |
|---|---|---|---|---|---|---|---|
| `no_augmentation` | 0.9586 ± 0.0008 | 0.9555 ± 0.0017 | 0.9463 ± 0.0022 | 0.9583 ± 0.0008 | 0.1835 ± 0.0005 | 0.1914 ± 0.0008 | 0.9623 ± 0.0005 |
| `joint_refined` | 0.9553 ± 0.0015 | 0.9535 ± 0.0010 | 0.9407 ± 0.0040 | 0.9550 ± 0.0016 | 0.2026 ± 0.0005 | 0.2611 ± 0.0008 | 0.9446 ± 0.0010 |
| `uniform_p010` | 0.9545 ± 0.0013 | 0.9526 ± 0.0015 | 0.9396 ± 0.0031 | 0.9542 ± 0.0014 | 0.2016 ± 0.0018 | 0.2644 ± 0.0001 | 0.9431 ± 0.0010 |
| `uniform_p020` | 0.9496 ± 0.0006 | 0.9477 ± 0.0022 | 0.9339 ± 0.0020 | 0.9491 ± 0.0008 | 0.2187 ± 0.0012 | 0.3152 ± 0.0008 | 0.9297 ± 0.0003 |
| `uniform_p025` | 0.9477 ± 0.0012 | 0.9467 ± 0.0024 | 0.9314 ± 0.0038 | 0.9472 ± 0.0014 | 0.2280 ± 0.0017 | 0.3363 ± 0.0015 | 0.9232 ± 0.0007 |

Across every recipe, macro precision exceeds macro recall, and macro-F1 sits below weighted F1. Both patterns suggest the weaker classes are the rarer ones, which are being under-recalled. (Per-class results aren't in these files, so this is an inference, not something verified.)

![Validation metrics grouped](figures/05_metrics_grouped.png)

## Learning curves

![Learning curves](figures/03_learning_curves.png)

Every recipe is still improving at epoch 3 (best epoch = 3 in all 15 runs), so none had converged.

## Effect of augmentation strength

![Augmentation strength vs macro-F1](figures/06_aug_strength_trend.png)

Among the uniform recipes, macro-F1 falls steadily as the probability rises: 0.9500 (0.00) → 0.9447 (0.10) → 0.9392 (0.20) → 0.9370 (0.25).

## Significance tests

Welch's two-sample t-test on per-seed macro-F1, **no multiple-comparison correction**. With n = 3 per group these p-values are indicative only.

Versus `no_augmentation`:

| Recipe | Difference in mean F1 | Welch t | p-value |
|---|---|---|---|
| `joint_refined` | -0.0044 | -2.48 | 0.113 |
| `uniform_p010` | -0.0053 | -4.12 | 0.032 |
| `uniform_p020` | -0.0108 | -9.61 | 0.003 |
| `uniform_p025` | -0.0130 | -8.34 | 0.007 |

`joint_refined` vs the uniform recipes:

| Compared with | Difference in mean F1 | Welch t | p-value |
|---|---|---|---|
| `uniform_p010` | +0.0009 | 0.43 | 0.690 |
| `uniform_p020` | +0.0064 | 3.23 | 0.043 |
| `uniform_p025` | +0.0086 | 3.81 | 0.020 |

## Takeaways

- No augmentation scored highest and was the most stable across seeds.
- The drop from `no_augmentation` to `uniform_p010`, `uniform_p020` and `uniform_p025` is larger than seed noise. The drop for `joint_refined` (p = 0.11) is **not** clearly distinguishable from noise at n = 3.
- `joint_refined` and `uniform_p010` are statistically indistinguishable (p = 0.69).
- `joint_refined` is modestly better than `uniform_p020` and `uniform_p025` (p = 0.04 and 0.02, uncorrected, so treat as suggestive).

## Caveats

- Best epoch was the last epoch (3) for every run, so no run had converged. Training only a linear head for 3 epochs probably favors no augmentation, since augmented data is harder to fit in a short schedule. Longer training could change the ranking.
- Only 3 seeds per recipe, so std, CI and p-values are rough.
- All scores are validation scores used for model comparison. No test-set numbers are reported here.

## Experimental setup

| Item | Value |
|---|---|
| Backbone | ResNet-50, frozen, eval mode |
| Trainable parameters | `fc` layer only |
| Epochs | 3 |
| Seeds | 42, 123, 2026 |
| Train rows | 37,751 |
| Validation rows | 7,976 |
| Test data used | No |
| Split SHA-256 | `e3e2d22d67a94bdc7d38763ca88fea372f2513907b4bb7c67a5d825f5797575a` |
| Hardware | Tesla T4 (CUDA 12.8) |
| Software | Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.1.3 |

All five experiments share the same split hash, so they are directly comparable.

## Augmentation recipes

Each value is the probability of applying that transform to a training image.

| Recipe | flip | rotation | colour | blur | noise | cutout | background |
|---|---|---|---|---|---|---|---|
| `no_augmentation` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `uniform_p010` | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| `uniform_p020` | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| `uniform_p025` | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| `joint_refined` | 0 | 0 | 0.25 | 0.10 | 0.25 | 0.25 | 0.10 |

## Repository layout

Analysis outputs (this README's tables and plots) live alongside the raw archives:

```
README.md
macro_f1_statistics.csv       # descriptive stats table above
all_metrics_mean_std.csv      # mean/std for every metric
welch_vs_no_aug.csv           # significance tests vs no_augmentation
figures/                      # PNG plots referenced above
```

Raw experiment archives:

Each archive holds one recipe and follows the same structure:

```
<experiment_dir>/
├── frozen_summary.json          # setup, environment, mean/std and per-seed scores
├── seed_results.csv             # recipe, seed, best epoch, best val macro-F1
└── <recipe>_seed<N>/            # one folder per seed
    ├── history.csv              # per-epoch train/val metrics
    ├── run_summary.json         # best epoch, best metrics, augmentation probabilities
    └── best_fc_checkpoint.pt    # weights of the trained fc layer (~315 KB)
```

| Archive | Top-level folder | Recipe |
|---|---|---|
| `frozen_no_aug_results.zip` | `frozen_no_augmentation/` | `no_augmentation` |
| `frozen_p010_results.zip` | `frozen_resnet50_runs/` | `uniform_p010` |
| `frozen_p020_results.zip` | `frozen_uniform_p020/` | `uniform_p020` |
| `frozen_p025_results.zip` | `frozen_uniform_p025/` | `uniform_p025` |
| `frozen_joint_refined_results.zip` | `frozen_joint_refined/` | `joint_refined` |

Note: the `p010` archive's folder is named `frozen_resnet50_runs/` (an earlier naming convention), but it contains only `uniform_p010` runs.

## File formats

**`history.csv`** columns: `recipe_name, seed, epoch, train_loss, train_accuracy, val_loss, val_accuracy, val_macro_precision, val_macro_recall, val_macro_f1, val_weighted_f1`

**`seed_results.csv`** columns: `recipe_name, seed, best_epoch, best_validation_macro_f1`

**`best_fc_checkpoint.pt`** contains only the classifier head, not the backbone. To reuse it, load a standard pretrained ResNet-50, freeze it, replace `model.fc` with a layer of matching shape, and load this state dict into it.

## Reproducing the aggregate table

```python
import json, glob

for path in sorted(glob.glob("*/frozen_summary.json")):
    s = json.load(open(path))
    for name, r in s["recipes"].items():
        print(f"{name:18s} {r['mean_best_val_macro_f1']:.4f} ± {r['sample_std']:.4f}")
```

## Suggested next steps

- Train for more epochs (or unfreeze later layers) to check whether augmentation helps once the model is no longer undertrained.
- Add more seeds to tighten the comparison between `joint_refined` and `uniform_p010`.
- Evaluate only the final chosen recipe on the held-out test set.
