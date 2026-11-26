# Calibration of ResNet-50 and MobileNetV3-Large

This section covers how well the models' confidence matches how often they are right, and the temperature and "unsure" rule derived from it. Both models use the final recipe (`uniform_p010`, full fine-tuning, 3 seeds: 42, 123, 2026).

**Checkpoints (all 6 final models):** https://kaggle.com/datasets/3c624b2b02c92dff7c68454d82fe0d51e0685fb12dc2fafeece8f1257eb13bfd
The files are not stored in GitHub because of its 100 MB limit.

## Method

1. **Validation logits.** `src/save_val_logits.py` runs each checkpoint once on the 7,976 validation images (same preprocessing as training: PIL bilinear resize to 224x224, ImageNet mean and standard deviation) and saves the raw scores to `results/calibration/val_logits/`. Reproduced validation macro-F1: ResNet-50 0.9836 / 0.9785 / 0.9819, MobileNetV3 0.9666 / 0.9771 / 0.9777 (seeds 42 / 123 / 2026). The MobileNetV3 values match the training logs within 0.0001. The test set is not used anywhere in this section.
2. **Half split.** Validation is split 50/50, stratified by class, with random seed 42 (`results/calibration/val_half_split.csv`). Temperature and the "unsure" threshold are fitted on the first half and scored on the second half.
3. **Reliability diagram and Expected Calibration Error (ECE).** Predictions go into 10 confidence bins. In each bin, average confidence is compared with accuracy. ECE is the gap per bin, weighted by the number of images in the bin.
4. **Temperature scaling.** One number T per checkpoint, found by minimising negative log-likelihood on the first half. The logits are divided by T before the softmax. Predictions do not change.
5. **PlantDoc confidence.** Confidence and accuracy of the 6 PlantDoc prediction files (241 images each) from `experiments/plantdoc-field-test/`.
6. **"Unsure" rule.** The threshold is the lowest confidence at which at least 99.5% of kept predictions are correct (after temperature scaling), chosen on the first half and checked on the second.

## Results

**Lab validation (second half, mean of 3 seeds)**

| Model | Accuracy | ECE before | ECE after | Mean T (range) |
|---|---|---|---|---|
| ResNet-50 | 0.9883 | 0.0049 | 0.0044 | 1.11 (0.93 to 1.30) |
| MobileNetV3-Large | 0.9827 | 0.0066 | 0.0040 | 0.90 (0.87 to 0.94) |

Both models are already well calibrated on lab images: ECE is under 1% before scaling. Temperatures fall on both sides of 1, so there is no single direction of over- or under-confidence. Per-seed values are in `calibration_per_seed.csv`, and the fitted T per checkpoint is in `temperature_values.json`.

**PlantDoc field images (241 images, mean of 3 seeds)**

| Model | Accuracy | Mean confidence | ECE |
|---|---|---|---|
| ResNet-50 | 0.31 | 0.70 | 0.39 |
| MobileNetV3-Large | 0.24 | 0.62 | 0.39 |

On field images both models report much higher confidence than their accuracy, and confidence separates right from wrong answers only weakly (ResNet-50 averages 0.79 when correct and 0.66 when wrong). The calibration found on lab images therefore holds only for lab-like images. Temperature scaling is not applied to PlantDoc, because T comes from lab validation and the PlantDoc files contain confidence only.

**"Unsure" rule (target: 99.5% accuracy on kept predictions)**

| Model | Threshold (mean of seeds) | Lab images kept | Accuracy on kept | Wrong answers flagged unsure |
|---|---|---|---|---|
| ResNet-50 | 0.81 | 97.7% | 99.55% | 62% |
| MobileNetV3-Large | 0.80 | 96.8% | 99.6% | 74% |

Rule for the app: if the calibrated confidence is below 0.80, show "unsure". On PlantDoc the same rule keeps 39% of images at 41% accuracy (ResNet-50) and 28% at 32% accuracy (MobileNetV3), so the rule is reliable for lab-like images but does not make field predictions safe (`unsure_threshold_on_plantdoc.csv`).

## Files

| Path | What it is |
|---|---|
| `src/save_val_logits.py` | Runs the checkpoints on validation and saves logits (Kaggle, GPU) |
| `src/calibration.py` | Half split, ECE, reliability diagrams, temperature scaling |
| `src/plantdoc_confidence.py` | PlantDoc confidence and accuracy, histogram and reliability plot |
| `src/unsure_threshold.py` | "Unsure" threshold chosen on validation |
| `results/calibration/reliability_diagrams.png` | Reliability diagrams, both models, before and after scaling |
| `results/calibration/calibration_summary.csv`, `calibration_per_seed.csv` | ECE, negative log-likelihood, T |
| `results/calibration/temperature_values.json` | Fitted T per checkpoint |
| `results/calibration/val_half_split.csv` | Which validation images are in each half |
| `results/calibration/plantdoc_confidence.png`, `plantdoc_calibration_summary.csv` | PlantDoc confidence results |
| `results/calibration/unsure_threshold_summary.csv`, `unsure_threshold_on_plantdoc.csv` | "Unsure" thresholds |
| `results/calibration/val_logits/` | Validation logits and labels (6 checkpoints) |

## Reproduce

From the repository root:

```
python src/calibration.py
python src/plantdoc_confidence.py
python src/unsure_threshold.py
```

These read `results/calibration/val_logits/` and `experiments/plantdoc-field-test/` and write to `results/calibration/`. Run `src/save_val_logits.py` on Kaggle only to regenerate the logits.
