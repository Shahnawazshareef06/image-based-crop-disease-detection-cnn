# Understanding the output files

How to read each file, what the numbers mean, and what they say about focal loss.

## 1. Metrics used

- **Validation macro-F1:** F1 score for each of the 38 classes, then a plain average. Every class counts equally, so a small class matters as much as a large one. This is the main metric because the question is about imbalance.
- **Validation accuracy:** share of validation images classified correctly. Large classes dominate it.
- **Seed:** a number that fixes random initialization and data order. Three seeds (42, 123, 2026) show how much results move by chance.
- **Std (standard deviation):** how far the three seed results spread around their mean. Used here as the yardstick for "is this difference real?". A difference smaller than the std is treated as no clear difference.
- **Best epoch:** the epoch (1 to 10) with the highest validation macro-F1 in that run. All reported scores come from this epoch.

## 2. Per-run folders (`focal_runs_*/<loss>_seed<N>/`)

One folder per loss and seed (9 per model).

- `history.csv`: training loss, validation accuracy and validation macro-F1 for every epoch. Use it for learning curves and to see whether training was still improving.
- `per_class_f1.csv`: F1 for each of the 38 classes at the best epoch.
- `run_summary.json`: settings, trainable parameter count, environment and the result row. The script reads it to skip finished runs.

## 3. Folder-level files

- `run_config.json`: the exact settings of the whole run (model, recipe, gammas, seeds, epochs, split hash, `test_data_used: false`). Proof of what was run.
- `per_seed_results.csv`: one row per run with best epoch, macro-F1, accuracy and F1 for the five smallest classes.
- `comparison_table.csv`: one row per loss, averaged over seeds: mean and std of macro-F1, F1 of each of the five smallest classes, and their mean (`f1_five_smallest_mean`).
- `conclusion_draft.txt`: auto-generated sentences comparing each focal loss with cross-entropy. It calls a difference "within the seed-to-seed spread" when it is no larger than the larger of the two stds. It is a draft only; edit before submitting.

## 4. Merged files

- `all_models_comparison.csv`: the two `comparison_table.csv` files stacked, with a `model` column.
- `all_models_per_seed_results.csv`: the two `per_seed_results.csv` files stacked.

## 5. The three 38-class files

These come from the per-run `per_class_f1.csv` files.

- **`all_38_classes_per_seed.csv`** (the `per_seed` sheet): 18 rows (2 models x 3 losses x 3 seeds). Columns: model, loss, seed, best epoch, macro-F1, accuracy, then one F1 column per class. This is the raw data; the other two files are summaries of it.
- **`all_38_classes_mean.csv`** (`mean_f1_by_class`): 38 rows, one per class. Six columns of seed-averaged F1 (one per model and loss) and four "minus_CE" columns. A minus_CE column is focal F1 minus cross-entropy F1 for that class: **positive means focal loss helped that class, negative means it hurt.**
- **`all_38_classes_std.csv`** (`std_by_class`): the seed-to-seed standard deviation of each cell in the mean file. A 0 means all three seeds gave the same F1. Use it to judge whether a minus_CE value is bigger than noise.

Std is the sample standard deviation over 3 seeds, so it is itself a rough estimate.

## 6. What the results show

### Overall (macro-F1, mean of 3 seeds)

| Model | Cross-entropy | Focal gamma 1 | Focal gamma 2 |
|---|---|---|---|
| ResNet-50 | 0.9708 | 0.9712 (+0.0004) | 0.9709 (+0.0001) |
| ViT-Tiny | 0.9555 | 0.9541 (-0.0014) | 0.9519 (-0.0036) |

- **ResNet-50:** differences are 0.0001 to 0.0004, much smaller than the seed spread (0.0004 to 0.0010). Treat as a tie. Seed by seed, gamma 1 was higher in two seeds and lower in one, with differences of at most 0.0016.
- **ViT-Tiny:** focal loss was lower in all three seeds for both gammas, and the drop grows with gamma. Gamma 2 (-0.0036) is about 2.5 times the std, so this one is likely real; gamma 1 is borderline. With only 3 seeds, "likely" is as strong as the wording should get.
- **Model gap:** ResNet-50 beats ViT-Tiny by about 0.015 macro-F1 under cross-entropy. That is far larger than any loss effect, so the choice of model matters more than the choice of loss here.

### Per class (mean file)

Number of classes where focal loss beat / lost to cross-entropy (the rest tied):

| | Gamma 1 | Gamma 2 |
|---|---|---|
| ResNet-50 | 22 better, 7 worse | 17 better, 12 worse |
| ViT-Tiny | 11 better, 22 worse | 7 better, 24 worse |

- ResNet-50 looks slightly positive class by class, but the average gain is only +0.0004 and +0.0001, so it is many tiny wins and a few losses, mostly inside the noise.
- Largest ResNet-50 gains: Corn Cercospora/Gray leaf spot (+0.0135, gamma 1), Potato Early blight (+0.0096, gamma 2). Largest loss: **Potato healthy (-0.0255)** for both gammas.
- Largest ViT-Tiny losses at gamma 2: Tomato Septoria leaf spot (-0.0196), Tomato Early blight (-0.0179), Grape Esca (-0.0146).
- Hardest classes under every loss: Corn Cercospora/Gray leaf spot, Tomato Early blight, Tomato Target spot. These look like hard-to-tell-apart diseases (visually similar tomato and corn leaf spots), not like a rarity problem. Focal loss does not fix them.
- Easiest: Corn Common rust, Raspberry healthy, Orange Huanglongbing and others at F1 of 1.0 or close to it, identical across losses. Nothing to gain there.

### The imbalance question

The five smallest training classes are Potato healthy (105), Apple Cedar apple rust (192), Raspberry healthy (248), Peach healthy (251), Tomato mosaic virus (257).

| Model | Cross-entropy | Focal gamma 1 | Focal gamma 2 |
|---|---|---|---|
| ResNet-50 | 0.9749 | 0.9701 | 0.9677 |
| ViT-Tiny | 0.9716 | 0.9728 | 0.9714 |

- For ResNet-50, focal loss is lower on the small classes, the opposite of what it is meant to do. Most of that drop is Potato healthy.
- For ViT-Tiny, it is flat.
- Potato healthy is the most extreme case (105 training images) and its validation set is small (about 20 images, estimated), so its F1 jumps by several points on one or two images. Do not read much into a single class.
- The small classes were never the weak spot. Raspberry healthy is perfect, and the weakest classes are the confusable diseases above. The imbalance in PlantVillage is real but mild for this setup, so there is little for focal loss to fix.

## 7. Caveats before quoting these results

- **Not converged.** Best epoch is 10 in 11 of 18 runs and 8 or 9 in the other 7. Scores were still rising, so conclusions are for a 10-epoch head-only budget.
- **Only the last layer is trained.** This tells us nothing about focal loss with full fine-tuning.
- **3 seeds.** Enough to see large effects, not small ones.
- **Selection on validation.** The best epoch is chosen on the same data it is scored on, so scores are slightly optimistic. The comparison between losses is still fair because all use the same rule.
- **Focal loss without alpha.** Class weights were not used, so this does not test the full focal loss with class balancing.
- **MobileNetV3-Large** is not included; it has not been run.
