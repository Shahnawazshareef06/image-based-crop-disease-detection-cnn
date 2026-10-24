# Next week goals: TerraVista

Decide among yourselves who does what before starting the work.

1. **Three seeds for every result.** Run each experiment with the same 3 seeds so everyone's results can be compared. Report the mean and standard deviation of validation macro-F1. A difference smaller than the spread is not a result.
2. **Make the runs reproducible.** Commit the PyTorch `augmentation_pipeline.py`, the baseline script and the imbalance script as real files, with a requirements file and fixed seeds. One command must rerun each from a clean clone, with no private Kaggle data.
3. **Remove the test set results.** Delete the test metrics, test reports and test confusion matrices from the repo, and take the test numbers out of the READMEs. Make the scripts compute test scores only when a flag is passed, and use it once, for the final model.
4. **Augmentation on ResNet-50, judged on validation.** Compare no augmentation, flips and rotation, plus colour changes, plus cutout. Change one thing at a time, 3 seeds each, and report mean and standard deviation of validation macro-F1.
5. **Try focal loss for class imbalance.** Train with gamma 1 and gamma 2 and compare with plain loss. Report validation macro-F1 and the F1 of the 5 smallest classes, 3 seeds each.
6. **Two more models.** Train MobileNetV3 and ViT-Tiny with the same split and settings as ResNet-50. Report validation macro-F1, number of parameters, file size and time per image for each.
7. **Error analysis of all 3 models.** For ResNet-50, MobileNetV3 and ViT-Tiny, on validation, report the weakest classes and the most confused class pairs. Count how many validation images all three get wrong and how many only one gets wrong. Write a short summary of how the models differ and why.
8. **Choose the field-image source.** Decide where field photos will come from (a public set such as PlantDoc, or photos you collect), confirm you can download it, and list which of the 38 classes it covers.
9. **Data-efficiency run.** Train the frozen ResNet-50 on 1%, 5%, 10%, 25%, 50% and 100% of the training images (stratified, 3 seeds) using the existing `--train-frac` option, and report validation macro-F1 at each size.
10. **Tidy the repository.** One clean structure of your own design, with no person-named folders, no old copies, and plain file names. Add a README with the data, setup, how to run each step, the results table and the limitations. Each person commits under their own name and email.

## Things to try for speed

- **Save the features once.** With a frozen backbone and no augmentation the features never change. Save train, validation and test features to a file and reuse them for every frozen-head run (seeds, focal loss, imbalance, data-efficiency).
- **Speed up loading.** Try `--workers 4`, `persistent_workers=True`, `prefetch_factor=4`, and `torch.set_num_threads(1)` inside each worker.
- **Augment on the GPU.** Decode the images once into memory and apply flips, rotation, colour changes and cutout on the GPU in batches. This also makes fine-tuning practical.
