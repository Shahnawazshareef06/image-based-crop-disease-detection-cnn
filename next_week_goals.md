# Next week goals: TerraVista

Decide among yourselves who does what and note it in the README before starting the work.

1. **One duplicate rule for the whole dataset.** Apply a single rule to all images in one pass (not per crop group), check a few samples of flagged pairs by eye, and log what was removed and why.
2. **Tune the duplicate threshold.** Treat the Hamming distance as a hyperparameter: try several values, and choose using what the flagged pairs look like and how many images each value removes, not a default.
3. **Remove near-copies between train and test.** Make sure no validation or test image has a near-identical training image, and report a leakage check that shows it.
4. **One split, reproducible from the repo.** Build the split from the cleaned data with a script, a fixed seed and stratification, and commit the split file. It must not depend on private Kaggle inputs.
5. **One framework.** Use PyTorch throughout, with the augmentation ported into the training transforms and applied to the training set only.
6. **Tune the augmentation.** Treat cutout percentage and the other augmentation strengths as hyperparameters: try a few values, pick them on validation results, and check a few samples by eye that the lesion stays visible.
7. **Metrics.** One module with accuracy, macro-F1, per-class F1 and a confusion matrix, with a small test that checks it.
8. **Frozen ResNet-50 baseline.** Train a new 38-class head on the frozen backbone and report validation results (accuracy, macro-F1, per-class F1). Do not use the test set yet.
9. **Class imbalance.** Try class weights or a weighted sampler and compare with the plain baseline on the smallest classes.
10. **Tidy the repository.** Organise it into a clean, sensible structure of your own design, with no folders named after people, no duplicate files, and plain file names.
11. **README and requirements.** Cover how to get the data, set up, run each step, the baseline results, known limitations, and who did what.
12. **Working practice.** Every member commits their own work with a meaningful message, with no upload-and-delete churn.
