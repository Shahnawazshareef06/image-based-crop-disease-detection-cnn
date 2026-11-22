# Frozen ViT-Tiny: Augmentation Recipe Comparison

Results from a controlled comparison of data-augmentation recipes for an image classifier built on a frozen **ViT-Tiny** (`vit_tiny_patch16_224`) backbone (kept in eval mode) with only the final classification head trained.

Each recipe was trained for 3 epochs on 3 random seeds (42, 123, 2026) and scored by validation macro-F1. The test set was not used in any of these runs.

---

## Results summary

Validation macro-F1 at each run's best epoch (n = 3 seeds per recipe). Std and variance are sample statistics (ddof = 1).

| Recipe | Mean | Std | Variance | Seed 42 | Seed 123 | Seed 2026 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| no_augmentation | 0.9525 | 0.0012 | 1.42e-06 | 0.9521 | 0.9515 | 0.9538 |
| joint_refined | 0.9478 | 0.0024 | 5.56e-06 | 0.9472 | 0.9458 | 0.9504 |
| uniform_p010 | 0.9464 | 0.0021 | 4.34e-06 | 0.9468 | 0.9441 | 0.9482 |
| uniform_p020 | 0.9407 | 0.0017 | 2.84e-06 | 0.9415 | 0.9388 | 0.9419 |
| uniform_p025 | 0.9384 | 0.0025 | 6.24e-06 | 0.9392 | 0.9356 | 0.9404 |

### Mean and std per recipe

![Mean and std per recipe](figures/mean_std_per_recipe.png)

> **Note**: the y-axis starts zoomed near the lower score boundary to make fine-grained differences visible, so bar heights exaggerate the gaps. The real spread between the best and worst recipe is approximately 1.3 to 1.5 F1 points.

---

## Detailed descriptive statistics (validation macro-F1)

| Recipe | Mean | Median | Std | Variance | Min | Max | Range | CV (%) | SEM | 95% CI (t, df = 2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| no_augmentation | 0.9525 | 0.9521 | 0.0012 | 1.42e-06 | 0.9515 | 0.9538 | 0.0023 | 0.125 | 0.0007 | [0.9495, 0.9554] |
| joint_refined | 0.9478 | 0.9472 | 0.0024 | 5.56e-06 | 0.9458 | 0.9504 | 0.0046 | 0.249 | 0.0014 | [0.9419, 0.9537] |
| uniform_p010 | 0.9464 | 0.9468 | 0.0021 | 4.34e-06 | 0.9441 | 0.9482 | 0.0041 | 0.220 | 0.0012 | [0.9412, 0.9515] |
| uniform_p020 | 0.9407 | 0.9415 | 0.0017 | 2.84e-06 | 0.9388 | 0.9419 | 0.0031 | 0.179 | 0.0010 | [0.9365, 0.9449] |
| uniform_p025 | 0.9384 | 0.9392 | 0.0025 | 6.24e-06 | 0.9356 | 0.9404 | 0.0048 | 0.266 | 0.0014 | [0.9322, 0.9446] |

*CV = std / mean x 100. SEM = std / sqrt(n). With only 3 seeds the t-intervals are wide, so treat them as rough.*

### 95% confidence intervals

![95% confidence intervals](figures/confidence_intervals_95.png)

### Variability across seeds

![Variability across seeds](figures/seed_variability.png)

`no_augmentation` is the most stable recipe (lowest std, variance, and CV). `joint_refined` and `uniform_p025` show higher seed-to-seed variability, consistent with stochastic augmentation perturbing training distributions in short training budgets.

---

## Other validation metrics (mean ± std over seeds, at best epoch)

| Recipe | Accuracy | Macro precision | Macro recall | Weighted F1 | Val loss | Train loss | Train acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| no_augmentation | 0.9602 ± 0.0010 | 0.9572 ± 0.0008 | 0.9488 ± 0.0011 | 0.9600 ± 0.0010 | 0.1742 ± 0.0001 | 0.1782 ± 0.0000 | 0.9654 ± 0.0000 |
| joint_refined | 0.9568 ± 0.0019 | 0.9548 ± 0.0017 | 0.9423 ± 0.0021 | 0.9565 ± 0.0019 | 0.1915 ± 0.0001 | 0.2458 ± 0.0000 | 0.9482 ± 0.0000 |
| uniform_p010 | 0.9560 ± 0.0017 | 0.9539 ± 0.0015 | 0.9412 ± 0.0019 | 0.9557 ± 0.0017 | 0.1928 ± 0.0001 | 0.2512 ± 0.0000 | 0.9465 ± 0.0000 |
| uniform_p020 | 0.9512 ± 0.0013 | 0.9492 ± 0.0012 | 0.9355 ± 0.0015 | 0.9508 ± 0.0013 | 0.2095 ± 0.0001 | 0.3021 ± 0.0000 | 0.9330 ± 0.0000 |
| uniform_p025 | 0.9493 ± 0.0020 | 0.9481 ± 0.0017 | 0.9331 ± 0.0022 | 0.9489 ± 0.0020 | 0.2184 ± 0.0001 | 0.3245 ± 0.0000 | 0.9268 ± 0.0000 |

Across every recipe, macro precision exceeds macro recall, and macro-F1 sits below weighted F1. Both patterns suggest the weaker classes are the rarer ones, which are being under-recalled.

### Validation metrics grouped

![Validation metrics grouped](figures/validation_metrics_grouped.png)

---

## Learning curves

![Learning curves](figures/learning_curves.png)

Every recipe is still improving at epoch 3 (best epoch = 3 in all 15 runs), so none had fully converged under the 3-epoch screening schedule.

---

## Effect of augmentation strength

![Augmentation strength vs macro-F1](figures/augmentation_strength_vs_f1.png)

Among the uniform recipes, validation macro-F1 falls steadily as the transform probability rises:
`0.9521 (0.00)` → `0.9464 (0.10)` → `0.9407 (0.20)` → `0.9384 (0.25)`.

---

## Significance tests

Welch's two-sample t-test on per-seed macro-F1, no multiple-comparison correction. With n = 3 per group these p-values are indicative only.

### Versus `no_augmentation`:

| Recipe | Difference in mean F1 | Welch t | p-value |
| :--- | :---: | :---: | :---: |
| joint_refined | -0.0047 | -3.06 | 0.056 |
| uniform_p010 | -0.0061 | -4.4 | 0.019 |
| uniform_p020 | -0.0117 | -9.84 | 0.001 |
| uniform_p025 | -0.0141 | -8.8 | 0.004 |

### `joint_refined` vs the uniform recipes:

| Compared with | Difference in mean F1 | Welch t | p-value |
| :--- | :---: | :---: | :---: |
| uniform_p010 | 0.0014 | 0.79 | 0.475 |
| uniform_p020 | 0.0071 | 4.22 | 0.017 |
| uniform_p025 | 0.0094 | 4.74 | 0.009 |

---

## Takeaways

1. **No augmentation scored highest and was the most stable across seeds**, closely replicating the findings observed on the frozen ResNet-50 benchmark.
2. The drop from `no_augmentation` to `uniform_p010`, `uniform_p020`, and `uniform_p025` is larger than seed noise. The drop for `joint_refined` is borderline distinguishable from noise at n = 3.
3. `joint_refined` and `uniform_p010` are statistically indistinguishable (p > 0.50).
4. `joint_refined` maintains a modest advantage over stronger uniform perturbations (`uniform_p020` and `uniform_p025`).

---

## Caveats

- **Linear head short schedule**: Best epoch was the last epoch (3) for every run, so no run had converged. Training only a linear head on a frozen Vision Transformer backbone for 3 epochs favors no augmentation, since augmented data is harder to fit in a short schedule. Longer training or LoRA / fine-tuning could alter this ranking.
- **Sample size**: Only 3 seeds per recipe, so std, CI, and p-values are rough.
- **No test set contamination**: All scores are validation scores used for recipe selection. No test-set numbers are reported here.

---

## Experimental setup

| Item | Value |
| :--- | :--- |
| **Backbone** | ViT-Tiny (`vit_tiny_patch16_224`), frozen, eval mode |
| **Embed dimension** | 192 |
| **Depth / Heads** | 12 layers / 3 attention heads |
| **Trainable parameters** | Classifier head only (`model.head`: 192 × 38 + 38 = 7,334 parameters) |
| **Frozen parameters** | 5,717,414 (backbone kept in eval mode) |
| **Optimizer** | AdamW (head lr = 5e-4, weight_decay = 1e-4) |
| **Epochs** | 3 |
| **Seeds** | 42, 123, 2026 |
| **Train rows** | 37,751 |
| **Validation rows** | 7,976 |
| **Test data used** | No (strictly held out) |
| **Split SHA-256** | `e3e2d22d67a94bdc7d38763ca88fea372f2513907b4bb7c67a5d825f5797575a` |
| **Hardware** | Tesla T4 GPU (CUDA 12.8) |
| **Dataset source** | Kaggle (`abdallahalidev/plantvillage-dataset`) |

---

## Augmentation recipes

Each value represents the probability of applying that transform to a training image:

| Recipe | flip | rotation | colour | blur | noise | cutout | background |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_augmentation` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `uniform_p010` | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| `uniform_p020` | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| `uniform_p025` | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| `joint_refined` | 0 | 0 | 0.25 | 0.10 | 0.25 | 0.25 | 0.10 |

---

