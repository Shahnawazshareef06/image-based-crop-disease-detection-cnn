# Reproducible ResNet-50 + Augmentation Handoff

This folder is intended to be copied into the team's training repository. It records the exact model/training protocol used in the five-recipe validation comparison and provides importable Python modules. It does **not** contain PlantVillage image files.

## 1. Files

- `augmentation_pipeline.py` — the team's augmentation implementation, plus the five named, validation-ranked probability recipes.
- `terravista_resnet50_settings.py` — importable ResNet-50 architecture, optimizer, loss, AMP, seed, worker and DataLoader settings.
- `terravista_bootstrap.py` — optional first-import Kaggle Triton compatibility guard. Use it before importing `torch` if running in the affected Kaggle image.
- `terravista_training_example.py` — reusable train/evaluation functions and a one-(recipe, seed) runner.
- `terravista_dataset.py` — Dataset adapter that reads the bundled split, applies the same PIL resize and builds a resumable training-only GrabCut alpha-mask cache.
- `run_five_recipe_comparison.py` — command-line entry point for all five recipes and three seeds.
- `train_val_split.csv` — exact ordered train/validation manifest used by the experiment; test rows are deliberately excluded from this handoff file.
- `class_to_idx.json` — exact sorted class-to-index mapping.
- `reproducibility_manifest.json` — split hash, class counts, environment reference, experiment protocol, recipe scores and probability vectors.
- `requirements-core.txt` — non-PyTorch dependencies (unpinned minimums). The Kaggle PyTorch/TorchVision CUDA builds should be kept consistent with the versions recorded in the manifest.
- `freeze_environment.py` — run once in the training environment to write `requirements-lock.txt` (exact pinned versions) and `environment_report.json` (Python, OS, PyTorch/CUDA/GPU). Commit both with your results.
- `verify_handoff.py` — fast CPU smoke test for hashes, class mapping, model head/optimizer, and all five augmentation branches (no pretrained-weight download).
- `.gitignore` — excludes caches, checkpoints, and run outputs from version control.

## 2. Reproducibility contract

To reproduce or fairly compare a recipe, keep all of these fixed:

| Item | Required value |
|---|---|
| Dataset | PlantVillage `color` image directory |
| Split | bundled `train_val_split.csv` (same row order and labels) |
| Training rows | 37,751 |
| Validation rows | 7,976 |
| Classes | 38 |
| Class mapping | bundled `class_to_idx.json` (alphabetical class-name order) |
| Input | RGB, `uint8`, NHWC, resized to 224 × 224 before augmentation |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |
| Model | ImageNet-pretrained TorchVision ResNet-50; replace `fc` with 38 outputs |
| Fine-tuning | all ResNet-50 layers trainable |
| Optimizer | AdamW with two parameter groups |
| Backbone LR | `1e-5` |
| Classifier LR | `5e-4` |
| Weight decay | `1e-4` |
| Loss | `CrossEntropyLoss` |
| LR scheduler | none |
| Train / validation batch | 32 / 64 |
| Epochs | 3 per recipe and seed |
| Seeds | 42, 123, 2026 |
| AMP | CUDA FP16 autocast + GradScaler |
| cuDNN | deterministic = true; benchmark = false |
| Matmul precision | `high` |
| Augmentation | training only; validation uses all probabilities zero |

The original logged Kaggle runtime reported PyTorch `2.11.0+cu128`, TorchVision `0.26.0+cu128`, CUDA available, and Triton hidden because `torch.compile` was not used. Use the same environment when possible. The original notebook selected a Kaggle T4 ×2 accelerator, but its training device was `cuda`; do not assume both GPUs were used by the experiment.

## 3. Install / environment

Use the same PyTorch/TorchVision CUDA environment as the study where possible. In Kaggle, prefer the preinstalled compatible CUDA builds rather than blindly reinstalling CPU-only wheels.

Install the remaining packages if they are missing:

```bash
pip install -r requirements-core.txt
python freeze_environment.py   # records exact versions -> requirements-lock.txt, environment_report.json
```

To rebuild an environment exactly, install the committed `requirements-lock.txt` (keep the preinstalled CUDA builds of PyTorch/TorchVision: the study ran `torch 2.11.0+cu128` and `torchvision 0.26.0+cu128`).

If you encounter the same Kaggle Torch/Triton import problem, restart the kernel and put this at the very top of `train.py`, before any PyTorch import:

```python
from terravista_bootstrap import apply_kaggle_triton_guard
apply_kaggle_triton_guard()
```

Do not call it after `torch`, `torchvision`, or `augmentation_pipeline` has been imported. Set `TERRAVISTA_KEEP_TRITON=1` only if you intentionally want to retain Triton.

## 4. Dataset contract — this is the part each teammate must implement consistently

Read `train_val_split.csv`. It contains only `train` and `val` rows, in the exact order used by the experiment. It uses columns `path`, `class`, and `split`. Resolve each relative `path` under the PlantVillage `color` directory.

Build class IDs using the bundled mapping. Do not create a new mapping independently in each notebook. Check that every validation class occurs in the train mapping.

Your PyTorch Dataset must return:

- Train, no-background recipe: `(image_uint8, label_id)`.
- Train, when `p_background > 0`: `(image_uint8, label_id, foreground_alpha_uint8)`.
- Validation: `(image_uint8, label_id)` only.

`image_uint8` must be RGB `uint8` with shape `[224, 224, 3]` (NHWC per sample). `augmentation_pipeline.augment_batch_gpu()` converts it to normalized float NCHW `[B, 3, 224, 224]`. Do **not** apply ImageNet normalization again after this function.

Fixed transform strengths are: rotation ±15°, brightness 0.60–1.40 (including darkening), contrast 0.80–1.20, saturation 0.80–1.20, hue shift ±0.02, sensor-noise standard deviation 0.02, blur kernel 5×5, and cutout area 2–15%.

For recipes with `p_background > 0` (including `uniform_p010`, `uniform_p020`, `joint_refined`, and `uniform_p025`), supply one matching `uint8` alpha mask per training image. Mask value 255 keeps the plant; 0 replaces the background. Use the same GrabCut estimator from `augmentation_pipeline.estimate_foreground_alpha_from_rgb()` and cache masks in the same training-row order. Generate masks from training images only. Never pass random masks to validation.

## 5. Run the complete five-recipe comparison

The package includes a Dataset adapter and command-line runner. It builds or reuses a resumable **train + validation RGB image cache** (about 6.41 GiB) using the original notebook's exact cache fingerprint and PIL bilinear resize. When any selected recipe needs background replacement, it also builds/reuses a separate **training-only alpha-mask cache** (about 0.44 GiB) using the same GrabCut version and original mask-cache fingerprint. Both caches use memory mapping; the image cache preserves train rows first and validation rows second. Test rows are discarded before any test paths are resolved.

First run `python verify_handoff.py` to check the files and imports. Then run from this folder (or copy it into your project repository):

```bash
python run_five_recipe_comparison.py \
  --color-dir /path/to/PlantVillage/color \
  --split-csv ./train_val_split.csv \
  --output-dir ./terravista_runs \
  --cache-dir ./terravista_cache
```


If the prior Kaggle version's Output is attached as an input and contains the two matching cache files, add its `cache` directory as a read-only cache source:

```bash
python run_five_recipe_comparison.py \
  --color-dir /path/to/PlantVillage/color \
  --reuse-cache-dir /kaggle/input/notebooks/<owner>/<version>/terravista_augmentation_validation_study/cache
```

The cache names are fingerprinted from the bundled split and implementation versions. For this split they are `plantvillage_train_val_224_9fb74edc1aef.npy` and `foreground_alpha_train_112_c95c6139a9c9.npy`, matching the original notebook cache names. A source cache is reused only if its filename, shape and dtype match; it is never written to.

The full comparison runner requires CUDA, just like the original Kaggle study. The default run executes 5 recipes × 3 seeds × 3 epochs = 45 training epochs. It saves one best checkpoint, epoch history, and a run-summary JSON per recipe/seed, plus `seed_results.csv` and `recipe_comparison_summary.json`. Completed pairs are skipped on rerun only when their saved recipe, seed, probability dictionary, split hash, model settings, software/hardware environment and artifacts match. The first run may spend several minutes building the large caches; subsequent runs reuse them when the cache directory is preserved.

To run one recipe only (for example the best augmented candidate):

```bash
python run_five_recipe_comparison.py --color-dir /path/to/PlantVillage/color --recipes uniform_p010 --seeds 42 123 2026
```

## 6. Integrate the helpers into an existing `train.py`

If the teammate already has Dataset classes, they can call `run_one_recipe_seed()` directly. Their Dataset must implement the contract above.

```python
# Must be the first project import if using the Kaggle Triton workaround.
from terravista_bootstrap import apply_kaggle_triton_guard
apply_kaggle_triton_guard()

from torch.utils.data import DataLoader
from augmentation_pipeline import get_augmentation_recipe
from terravista_resnet50_settings import (
    SEEDS, COMPARISON_EPOCHS, seed_everything,
    make_dataloader_kwargs, build_resnet50, make_optimizer,
    make_criterion, make_grad_scaler, model_settings_dict,
)
from terravista_training_example import run_one_recipe_seed
from my_dataset import PlantVillageTrainDataset, PlantVillageValDataset

# Construct datasets from the bundled train_val_split.csv and class_to_idx.json.
train_dataset = PlantVillageTrainDataset(...)
val_dataset = PlantVillageValDataset(...)
class_to_idx = ...  # load the bundled class_to_idx.json

# Reproduce the preferred augmented recipe, using identical seeds.
for seed in SEEDS:
    result = run_one_recipe_seed(
        train_dataset,
        val_dataset,
        recipe_name="uniform_p010",
        seed=seed,
        class_to_idx=class_to_idx,
        output_dir="runs/terravista",
    )
    print(result["best_validation_macro_f1"])
```

The `run_one_recipe_seed()` helper calls `seed_everything(seed)` before constructing the model and loaders, uses the dedicated training DataLoader generator, and stores a checkpoint plus a run-summary JSON for each seed. To compare all recipes, loop over the five recipe names and then over `SEEDS`; do not change model settings between recipes.

## 7. Five supported recipes

| Recipe name | Meaning | Mean best validation macro-F1 | Std (3 seeds) | Difference vs no augmentation |
|---|---|---:|---:|---:|
| `no_augmentation` | all seven probabilities = 0 | 0.984665 | 0.0009 | — |
| `uniform_p010` | all seven probabilities = 0.10 | 0.984099 | 0.0022 | −0.0006 |
| `uniform_p020` | all seven probabilities = 0.20 | ≈ 0.983800 | 0.0016 | −0.0009 |
| `joint_refined` | flip 0, rotation 0, colour 0.25, blur 0.10, noise 0.25, cutout 0.25, background 0.10 | ≈ 0.983067 | 0.0024 | −0.0016 |
| `uniform_p025` | all seven probabilities = 0.25 | ≈ 0.982333 | 0.0025 | −0.0023 |

These means are from the matched five-recipe study: 3 seeds × 3 epochs, full train/validation splits; "best" is the best epoch per seed, and std is the sample standard deviation across the three seeds.

**How to read this table.** A difference smaller than the seed-to-seed spread is not a result. Every augmented recipe differs from no augmentation by less than the larger of the two spreads, so the five recipes are statistically tied. The ordering (no augmentation first, `uniform_p010` best among augmented recipes) is the preference order by mean score only. **Use `no_augmentation` as the default.** Treat any augmented recipe as optional, and call an improvement real only if it beats the 3-seed spread.

Some means were reconstructed from per-seed log values rounded to four decimals, so treat those displayed values as approximate. The original log notes that 12 of 15 runs reached their best validation macro-F1 in epoch 3; the ranking is a controlled screening result, not proof that all runs had converged. Validation here is in-distribution PlantVillage, which rarely rewards augmentation; robustness to field images needs a separate field-image test set.

### 7b. Earlier four-policy study

Before the probability study, four cumulative fixed policies were compared on the same split with the same seeds, model settings and 3 epochs (validation only):

| Policy | Settings | Mean best validation macro-F1 | Std | Difference vs no augmentation |
|---|---|---:|---:|---:|
| No augmentation | all zero | 0.98467 | 0.0009 | — |
| Flips + rotation | horizontal and vertical flips p = 0.5 each; rotation ±15° on every image | 0.97917 | 0.0019 | −0.0055 |
| + colour | previous + mild brightness/contrast/saturation (0.9–1.1) on every image | 0.98017 | 0.0012 | −0.0045 |
| + cutout | previous + cutout p = 0.25, max area 10% | 0.97977 | 0.0022 | −0.0049 |
| + background variation | previous + GrabCut background replacement p = 0.10 | 0.97997 | 0.0022 | −0.0047 |

These policies applied flips, rotation and colour to every image, and all four scored 0.0045–0.0055 below no augmentation, which is larger than the seed spread. **These four rows are not reproducible with this folder.** They were produced by an older mode-based augmentation function in the team notebook `wk5notebook.ipynb` (not included here), whose strengths are milder than this folder's fixed ranges (colour 0.9–1.1 with no hue shift, cutout up to 10% area). `augment_batch_gpu()` in this folder uses the wider fixed strengths listed in Section 4, so it reproduces the five recipes in Section 7 but not these four policies exactly.

## 8. What to save for reproducibility

For every run, preserve:

1. the bundled split manifest and class map;
2. recipe name and all seven `p_*` values;
3. seed and epoch-by-epoch train/validation metrics;
4. best validation checkpoint and best epoch;
5. model/optimizer/AMP settings and Python/PyTorch/TorchVision/CUDA/GPU versions;
6. preprocessing and mask-cache algorithm/version.

Do not use test metrics for recipe selection. Report the validation split and training budget with every score.

## 9. Reproducibility limitation

This setup makes the experiment reproducible in protocol, data order, seeds and configuration. It does not guarantee bit-for-bit identical scores across different GPU models, CUDA/cuDNN builds, PyTorch/TorchVision releases, or differing Dataset implementations. Use the same hardware/software stack and exactly the same preprocessing, class map and mask cache to minimize variation.
