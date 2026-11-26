# Imbalance Handling for PlantVillage Classification

Compares six ways of handling class imbalance on top of a frozen ResNet-50 baseline for the PlantVillage "color" dataset (38 classes).

The imbalance is large: the biggest training class (Orange, Huanglongbing) has 3,841 images and the smallest (Potato, healthy) has 105, a ratio of about 36x.

## What it does

1. Reads `split.csv` (columns `path, class, split`). The split is fixed and never modified here.
2. Extracts a 2,048-number feature vector for every image with an ImageNet-pretrained ResNet-50 (`IMAGENET1K_V2`). The backbone is frozen and in eval mode. Images are resized to 224x224 and normalised, with no augmentation.
3. Caches the features in `features_cache.pt`, so later runs skip extraction.
4. Trains the same classifier head six times on the cached features. Only the imbalance technique changes.
5. Selects the best epoch of each run by validation macro F1, picks the overall winner by validation macro F1, and reports test metrics for all variants. Test data is never used for selection.

Classifier head: `Dropout(0.2)` + `Linear(2048, 38)`. Optimiser AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, 30 epochs, batch size 256, seed 42.

## Variants

| Variant | Technique |
|---|---|
| `baseline` | plain cross-entropy (control) |
| `weighted_loss_inverse` | class weight proportional to 1 / count |
| `weighted_loss_sqrt_inverse` | class weight proportional to 1 / sqrt(count) |
| `weighted_loss_effective_number` | class weight (1 - beta) / (1 - beta^count), beta = 0.999 |
| `sampling_sqrt` | images sampled with probability proportional to count^-0.5 |
| `sampling_balanced` | every class gets an equal share of samples |

Class weights are rescaled to average 1. Sampling variants draw as many images per epoch as there are training images.

## Results

Split sizes: train 37,751, val 7,976, test 7,978.

| Variant | Best epoch | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 |
|---|---|---|---|---|---|
| baseline | 27 | 0.9739 | **0.9689** | 0.9695 | 0.9630 |
| weighted_loss_inverse | 28 | 0.9718 | 0.9635 | 0.9690 | 0.9626 |
| weighted_loss_sqrt_inverse | 26 | 0.9737 | 0.9670 | 0.9699 | 0.9635 |
| weighted_loss_effective_number | 27 | 0.9734 | 0.9660 | 0.9690 | 0.9617 |
| sampling_sqrt | 27 | 0.9733 | 0.9673 | 0.9693 | 0.9629 |
| sampling_balanced | 30 | 0.9729 | 0.9659 | 0.9685 | 0.9622 |

The winner by validation macro F1 is `baseline`. The week 3 baseline run reported test macro F1 0.9635; the control rerun here gives 0.9630 because the random-number state is not identical.

### Rare classes (test set)

| Class | Train images | Test images | Baseline recall | Best variant recall |
|---|---|---|---|---|
| Potato healthy | 105 | 23 | 0.739 | 0.826 (weighted_loss_inverse) |
| Corn Cercospora leaf spot | 359 | 77 | 0.844 | 0.870 (several) |

Weighted loss raised recall on Potato healthy by one or two images, but precision fell (from 1.00 to about 0.86 with inverse weighting), so its F1 did not improve (0.850 to 0.844).

### Findings

- All six variants land within 0.002 test macro F1 of each other. That is about the same size as the gap between two runs of the same baseline, so the differences are not meaningful.
- The weakest classes are mostly look-alike confusions, not small classes. Tomato Early blight has 700 training images and is the weakest class (F1 0.827); its errors go to other tomato diseases. Re-weighting does not fix that.
- With the backbone frozen, imbalance handling only moves the decision boundaries of the head. It cannot change the features.

## Run

Requirements: `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `opencv-python`, `pillow`, `tqdm`.

`imbalance_experiments.py` imports `train_baseline.py` and `augmentation_pipeline.py`, so keep all three in the same folder.

```
python imbalance_experiments.py --csv split.csv --root /path/to/plantvillage/color --out results_imbalance --workers 4
```

- `--root` is the folder that the paths in `split.csv` are relative to.
- The first run downloads the ResNet-50 weights (internet needed) and extracts features. This took about 24 minutes on a Kaggle GPU. Later runs load `features_cache.pt` and finish in minutes.
- On Kaggle, run `imbalance-handling.ipynb`, which copies the input files, writes the script, runs a quick smoke test on synthetic features and then does the real run.

## Outputs (`results_imbalance/`)

- `imbalance_summary.csv`, `metrics.json`: one row per variant.
- `per_class_comparison.csv`: test F1 and recall per class for every variant.
- `imbalance_comparison.png`: macro F1 per variant, plus per-class change for the winner against the baseline.
- `winner_confusion_matrix.png`: test confusion matrix of the winner.
- One folder per variant with `head_best.pt`, `training_curve.csv`, `val_per_class.csv` and `test_per_class.csv`.

## Limitations

- Single seed (42). Differences this small need several seeds before any variant can be called better or worse.
- The Potato healthy test set has 23 images, so one image changes its recall by about 4%.
- No augmentation and a frozen backbone. Fine-tuning part of the backbone, or combining these methods with augmentation, was not tested.
- The split file comes from the team's split step. This repo does not check it for near-duplicates between train and test.
