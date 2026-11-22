# Member 3 handover: PlantVillage test evaluation and near-duplicate check

Everything here was produced by ONE run of the notebook. The test set was used exactly once per model per seed
(2 models x 3 seeds = 6 evaluations). Nothing was changed after seeing test results.

## Main numbers (test set, 7978 images, 38 classes)
| Model | Test macro-F1, mean ± SD over 3 seeds (95% bootstrap CI of the seed mean) | After removing near-duplicates |
|---|---|---|
| ResNet-50 | 0.9821 ± 0.0022 (95% CI 0.9787–0.9848) | 0.9818 ± 0.0022 |
| MobileNetV3-Large | 0.9644 ± 0.0093 (95% CI 0.9592–0.9686) | 0.9640 ± 0.0094 |

Paired comparison on the same resampled test images (ResNet-50 minus MobileNetV3): +0.0178, 95% CI +0.0136 to +0.0220
(the interval excludes zero).

Near-duplicates: 353 of 7978 test images (4.4%) have a training image with cosine similarity >= 0.95
(ImageNet ResNet-50 features). The threshold was fixed before any test result. The perceptual-hash signal flagged 53 images, 13 by both.

## Files
- `test_results_summary.csv`, `test_results_per_seed.csv`: the numbers above, per seed and averaged.
- `bootstrap_confidence_intervals.csv`, `paired_comparison.csv`: confidence intervals and the model-vs-model comparison.
- `per_class_test_f1.csv`: per-class F1 and number of test images (classes under 50 images are marked; their F1 is noisy).
- `predictions/test_predictions_<model>_seed<seed>.csv`: path, true/predicted class, confidence, near-duplicate flag for every test image.
- `predictions/test_logits_<model>_seed<seed>.npy`: raw model outputs, shape (7978, 38), same row order as `test_image_order.csv`.
- `near_duplicate/`: per-image flags with the nearest training image, summary, and a threshold-sensitivity table (information only).
- `validation_reproduction_check.csv`: proof that the notebook reproduced the teammates' validation scores before touching test.
- `config_test_evaluation.json`: every setting, the near-duplicate rule, hashes of all checkpoints and the split file, software versions.
- `class_to_idx.json`, `test_image_order.csv`, `run_log.txt`.

## For Member 4 (PlantDoc field test)
Do not run the models on the lab images again. Read `predictions/test_predictions_<model>_seed<seed>.csv`, keep the classes that PlantDoc covers,
and compute lab macro-F1 on those classes only. Use `flag_near_duplicate == False` if you want the leakage-free lab score.

## Honest limits
- Only 3 seeds; the SD is a sample SD (n-1). The bootstrap interval reflects which test images were drawn, not extra seeds.
- Near-duplicate detection is a similarity rule, not proof. Different thresholds give different counts (see the sensitivity table).
- Several classes have very few test images (see `per_class_test_f1.csv`).
- Both models were fully fine-tuned with `uniform_p010`, 3 epochs; results do not apply to other training setups.
