# Next week goals: TerraVista

Decide among yourselves who does what before starting the work. All of these carry over from last week.

1. **Two more models.** Train MobileNetV3 and ViT-Tiny with the same split, seeds (42, 123, 2026) and number of epochs as the fine-tuned ResNet-50 you compare against. Report validation macro-F1 (mean and standard deviation), number of parameters, file size and time per image for all three models in one table.
2. **Error analysis of all 3 models.** On validation, report each model's weakest classes and most confused class pairs. Count how many validation images all three get wrong and how many only one gets wrong. Write a short summary of how the models differ and why.
3. **Check the focal-loss result.** Rerun cross-entropy and focal loss (gamma 2) for 10 epochs, and give cross-entropy two learning rates, to see whether the gain is more than a learning-rate or training-length effect. Use the same 3 seeds and report mean and standard deviation. If the gain disappears, say so.
4. **Data-efficiency run.** Train the frozen ResNet-50 on 1%, 5%, 10%, 25%, 50% and 100% of the training images (stratified, 3 seeds) and report validation macro-F1 at each size.
5. **Choose the field-image source.** Decide where field photos will come from (a public set such as PlantDoc, or photos you collect), confirm you can download it, and list which of the 38 classes it covers.
6. **Remove the test set results.** Delete the test metrics, test reports and test confusion matrices still in the repo (the Week 3 `Model traning` folder), and take test numbers out of the READMEs.
7. **Make every run reproducible.** Commit the scripts your READMEs tell people to run (for example `run_focal_loss.py`, `run_frozen_resnet50.py` and the training scripts for the new models). Each experiment must rerun with one command from a clean clone, and `verify_handoff.py` must pass.
8. **Tidy the repository.** One clean structure of your own design, with no person-named folders, no old copies, no stray placeholder files, and plain file names. Add a README with the data, setup, how to run each step, the results table and the limitations.

## Things to try for speed

- Reuse the decoded-image cache and the `terravista_*` scripts from `augmentation-study/reproducibility` for the new models, so images are decoded once.
- With a frozen backbone and no augmentation, save the features once and reuse them for every frozen-head run (for example data-efficiency).
