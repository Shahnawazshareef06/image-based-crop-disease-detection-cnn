# TerraVista – Data-Efficiency Experiments (PlantVillage, 38 classes)

This bundle holds the results of two data-efficiency studies, where a classifier is trained on
1 %, 5 %, 10 %, 25 %, 50 % and 100 % of the training set, each with 3 random seeds (42, 123, 2026).
All metrics are measured on the **validation split (7,976 images)**. The test set was not used.

The results are shipped as three zip archives:

| Archive | Size | What it contains |
|---|---|---|
| `terravista_finetune_p10_logs_predictions.zip` | 5.8 MB | Logs, per-epoch metrics, per-image validation predictions, summaries, plot, config and training script for the **ResNet-50 full fine-tuning** study (`uniform_p010` recipe) |
| `resnet50_p10_finetune_checkpoints.zip` | 264 MB | Best-epoch model weights (`best_model.pt`) for the three **100 %-data** runs only |
| `data_efficiency_results.zip` | 91 KB | Earlier study (11 Nov 2026) using the **`no_augmentation`** recipe: histories, summaries and plot (no predictions, logs or checkpoints) |

---

## 1. `terravista_finetune_p10_logs_predictions.zip`

```
runs_finetune_p10/
├── terravista_finetune_p10_config.json   # exact hyperparameters (config_sha256 is logged in every run)
├── train_runs.py                         # training/evaluation script that produced the runs
└── resnet50/
    ├── all_runs.csv                      # one row per run (18 rows)
    ├── summary_by_fraction.csv           # mean/sd/min/max over the 3 seeds, per fraction
    ├── data_efficiency_curve.png         # macro-F1 vs. training fraction
    └── frac<P>_seed<S>/                  # 18 run folders, e.g. frac10_seed42
        ├── train.log                     # full run log (config hash, environment, per-epoch lines)
        ├── history.csv                   # per-epoch train/val loss, accuracy, macro-F1, weighted-F1
        ├── run_summary.json              # best epoch, best val metrics, params, runtime, environment
        └── val_predictions.csv           # per-image predictions of the best epoch
```

`<P>` is the training percentage (1, 5, 10, 25, 50, 100) and `<S>` the seed.

### Training setup (from `terravista_finetune_p10_config.json`)

| Item | Value |
|---|---|
| Model | ResNet-50, torchvision ImageNet weights (`DEFAULT`), 23.59 M parameters, new 38-way head |
| Mode | Full fine-tuning, all layers trainable |
| Optimizer | AdamW, head LR 5e-4, backbone LR 1e-5, weight decay 1e-4, no scheduler |
| Loss / precision | CrossEntropy, FP16 AMP |
| Epochs / batch size | 3 epochs, train batch 32, val batch 64 |
| Input | 224 × 224 |
| Augmentation recipe | `uniform_p010` |
| Subsets | Stratified per seed; 100 % uses the full training set in original order |
| Model selection | Best epoch by validation macro-F1 (checkpoint and predictions come from that epoch) |
| Hardware | Tesla T4, PyTorch 2.11.0+cu128, Python 3.13.15 |

Train-set sizes: 377 / 1,887 / 3,772 / 9,438 / 18,874 / 37,751 images for 1 / 5 / 10 / 25 / 50 / 100 %.

### Results (validation, mean over 3 seeds)

| Fraction | n_train | Macro-F1 (mean ± sd) | Accuracy | Train time / run |
|---|---|---|---|---|
| 1 % | 377 | 0.087 ± 0.017 | 0.322 | 38 s |
| 5 % | 1,887 | 0.788 ± 0.011 | 0.868 | 59 s |
| 10 % | 3,772 | 0.908 ± 0.001 | 0.933 | 87 s |
| 25 % | 9,438 | 0.959 ± 0.001 | 0.968 | 168 s |
| 50 % | 18,874 | 0.977 ± 0.001 | 0.982 | 303 s |
| 100 % | 37,751 | 0.981 ± 0.003 | 0.987 | 573 s |

The config defines two thresholds for "enough data": absolute macro-F1 ≥ 0.88, or within 0.01 of the full-data result.
By these numbers, 10 % of the data already clears the absolute threshold (0.908), and 50 % is within 0.01 of the 100 % result (0.977 vs. 0.981).
At 1 % the model has only 3 epochs (~36 optimizer steps) and barely learns the head, so those numbers are not stable.

### File formats

- **`val_predictions.csv`**: columns `path, y_true, y_pred, confidence, true_class, pred_class` (7,976 rows per run). `path` is relative to the PlantVillage `color` directory (`<class>/<image>.JPG`).
- **`history.csv`**: one row per epoch with `arch, fraction, seed, epoch, epoch_seconds, train_loss, train_accuracy, val_loss, val_accuracy, val_macro_f1, val_weighted_f1`.
- **`run_summary.json`**: includes `config_sha256`, so each run can be matched to the config file.

> Note: `checkpoint_mb` is `null` in the summaries, even for the 100 % runs whose weights are shipped in the checkpoints archive.

---

## 2. `resnet50_p10_finetune_checkpoints.zip`

```
runs_finetune_p10/resnet50/
├── frac100_seed42/best_model.pt
├── frac100_seed123/best_model.pt
└── frac100_seed2026/best_model.pt
```

Each file is about 90 MB and is a **model-only** checkpoint (no optimizer state) from the best-validation epoch of the 100 % run.
Only 100 % runs have checkpoints (the script default is `--checkpoints full`).
The folder paths match those in the logs archive, so extracting both into the same directory merges them.

Loading sketch (assumes the file holds a `state_dict`; check the keys if it is wrapped):

```python
import torch
from torchvision import models

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 38)
state = torch.load("runs_finetune_p10/resnet50/frac100_seed42/best_model.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

---

## 3. `data_efficiency_results.zip`

An earlier data-efficiency study with the **`no_augmentation`** recipe (no augmentation, 3 epochs, 3 seeds, same six fractions).

```
data_efficiency_runs/
├── data_efficiency_config.json           # recipe, fractions, seeds, flags, environment
├── data_efficiency_summary.csv           # mean/std/min/max macro-F1 per fraction
├── per_run_results.csv                   # one row per run (18 rows, incl. total_steps)
├── data_efficiency_curve.png
└── no_augmentation_frac<P>_seed<S>/      # 18 folders: history.csv + run_summary.json
```

Config flags: `stratified: true`, `nested_subsets_per_seed: true`, `match_steps: false` (fewer images means fewer steps, not more epochs), `test_data_used: false`.

| Fraction | n_train | Macro-F1 (mean ± sd) | Final accuracy |
|---|---|---|---|
| 1 % | 377 | 0.061 ± 0.011 | 0.280 |
| 5 % | 1,887 | 0.507 ± 0.022 | 0.702 |
| 10 % | 3,772 | 0.735 ± 0.008 | 0.840 |
| 25 % | 9,438 | 0.867 ± 0.002 | 0.911 |
| 50 % | 18,874 | 0.918 ± 0.002 | 0.939 |
| 100 % | 37,751 | 0.950 ± 0.001 | 0.959 |

The `history.csv` files here also include macro precision and recall.

---

## Comparing the two studies

| Fraction | `no_augmentation` (study 3) | `uniform_p010` fine-tune (study 1) |
|---|---|---|
| 5 % | 0.507 | 0.788 |
| 10 % | 0.735 | 0.908 |
| 25 % | 0.867 | 0.959 |
| 50 % | 0.918 | 0.977 |
| 100 % | 0.950 | 0.981 |

Caveat: the two studies were run on different dates, and `data_efficiency_config.json` does **not** record the architecture or learning rates of the earlier study.
The difference may therefore come from more than the augmentation recipe, so treat it as indicative until the setup is confirmed.

## Reproducing

```bash
python train_runs.py --config terravista_finetune_p10_config.json \
    --color-dir <PlantVillage/color> --arch resnet50 \
    --fractions 0.01 0.05 0.1 0.25 0.5 1.0

# rebuild summary_by_fraction.csv / all_runs.csv from existing runs
python train_runs.py --config terravista_finetune_p10_config.json --summarize-only
```

`train_runs.py` imports helper modules that are **not** included in these archives (`terravista_bootstrap`, `augmentation_pipeline`, `terravista_dataset`, `terravista_resnet50_settings`), and it expects `train_val_split.csv` and `class_to_idx.json` next to the script.
The config records their SHA-256 hashes (`split_csv_sha256`, `class_map_sha256`) for verification.
