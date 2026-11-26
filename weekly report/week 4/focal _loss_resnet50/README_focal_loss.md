# Focal Loss vs Cross-Entropy on a Frozen ResNet-50 (PlantVillage)

Compares standard cross-entropy with focal loss (gamma 1 and gamma 2) for handling class imbalance on the PlantVillage "color" dataset (38 classes). Everything else is held fixed, so the loss function is the only difference between runs.

Focal loss multiplies each image's cross-entropy by `(1 - p)^gamma`, where `p` is the probability the model gives the correct class. Images the model already classifies confidently contribute almost nothing, and the training signal comes mostly from the hard ones. Gamma 0 is plain cross-entropy.

## Setup

| Item | Value |
|---|---|
| Model | ImageNet-pretrained ResNet-50, backbone frozen (eval mode), only the final linear layer (`fc`) trained |
| Augmentation recipe | `uniform_p010` (flip, rotation, colour, blur, noise, cutout and background replacement, each with probability 0.1) |
| Optimizer | AdamW, learning rate 5e-4 for the classifier, weight decay 1e-4, no scheduler |
| Training | 3 epochs, batch size 32, mixed precision |
| Seeds | 42, 123, 2026 |
| Data | `train_val_split.csv`: 37,751 train and 7,976 validation images |
| Test set | never loaded or evaluated |
| Model selection | best epoch by validation macro F1 |

The frozen setup and recipe follow the team's earlier frozen run. The cross-entropy rerun here reproduces it: 0.9452 +/- 0.0022 against the earlier 0.9447 +/- 0.0021 (seed 42 matches to four decimals).

## Results (validation, mean of 3 seeds)

| Loss | Macro F1 | Std | Accuracy |
|---|---|---|---|
| Cross-entropy | 0.9452 | 0.0022 | 0.9548 |
| Focal gamma 1 | 0.9480 | 0.0034 | 0.9569 |
| Focal gamma 2 | 0.9488 | 0.0021 | 0.9575 |

Per-seed macro F1:

| Loss | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---|---|
| Cross-entropy | 0.9453 | 0.9430 | 0.9473 |
| Focal gamma 1 | 0.9494 | 0.9442 | 0.9505 |
| Focal gamma 2 | 0.9498 | 0.9464 | 0.9502 |

### Five smallest classes (F1)

| Class | Train images | Cross-entropy | Focal gamma 1 | Focal gamma 2 |
|---|---|---|---|---|
| Potato healthy | 105 | 0.853 | 0.881 | 0.889 |
| Apple Cedar apple rust | 192 | 0.992 | 0.996 | 0.992 |
| Raspberry healthy | 248 | 0.987 | 0.990 | 0.990 |
| Peach healthy | 251 | 0.953 | 0.952 | 0.952 |
| Tomato mosaic virus | 257 | 0.942 | 0.929 | 0.935 |
| **Mean** | | **0.9453** | **0.9498** | **0.9517** |

## Conclusion

Focal loss with gamma 2 improved validation macro F1 from 0.9452 to 0.9488 (+0.0036), and it was better than cross-entropy for all three seeds (+0.0045, +0.0035, +0.0028). Gamma 1 was also better for all three seeds but by less (+0.0028 on average). A paired t-test over the three seeds gives p of about 0.02 for gamma 2 and about 0.08 for gamma 1, without correcting for the two comparisons.

The mean F1 of the five smallest classes rose from 0.9453 to 0.9517 with gamma 2. Most of that comes from Potato healthy (0.853 to 0.889). Tomato mosaic virus got slightly worse (0.942 to 0.935), and the other three classes barely changed.

The improvement is small and in a consistent direction, but it is suggestive, not conclusive.

## Limitations

- Three seeds and three epochs. The best epoch was the last one in every run, so training had not converged and the gap could change with longer training.
- The small classes have very small validation sets, so one or two images can move their F1 noticeably.
- Only the frozen setup was run. `--frozen` trains the last layer only; running without it fine-tunes all layers, which was not part of this experiment.
- Validation only. No test results are reported.

## Run

Needs a CUDA GPU. Put these in one folder: `run_focal_loss.py`, `augmentation_pipeline.py`, `terravista_bootstrap.py`, `terravista_dataset.py`, `terravista_resnet50_settings.py`, `terravista_training_example.py`, `train_val_split.csv`, `class_to_idx.json`, `reproducibility_manifest.json`.

```
python verify_handoff.py
python run_focal_loss.py --color-dir /path/to/PlantVillage/color --frozen --include-ce
python freeze_environment.py
```

- `--frozen` trains only the final layer. Without it, all layers train.
- `--include-ce` also trains cross-entropy with the same code, so all three losses are compared under identical conditions.
- The first run builds a decoded-image cache of about 7 GB. To reuse an existing one, add `--reuse-cache-dir <folder containing the plantvillage_train_val_224_*.npy file>`.
- Re-running the same command skips runs that already finished.
- The first run downloads the ResNet-50 ImageNet weights, so internet access is needed.

## Outputs (`focal_runs_frozen/`)

- `per_seed_results.csv`: one row per loss and seed, with validation macro F1, accuracy and the F1 of the five smallest classes.
- `comparison_table.csv`: mean and standard deviation per loss, and the five-smallest-classes F1.
- `conclusion_draft.txt`: auto-generated draft of the comparison.
- One folder per run with `history.csv`, `per_class_f1.csv` and `run_summary.json`.
- `requirements-lock.txt` and `environment_report.json` (from `freeze_environment.py`) record the package versions used.
