# Frozen ResNet-50 Baseline (No Augmentation)

## Overview

This part trains the baseline classifier for the TerraVista PlantVillage crop disease classification task. It uses PyTorch and torchvision.

A ResNet-50 pretrained on ImageNet is kept completely frozen and used as a feature extractor. A new linear classification head for 38 classes is trained on top of it, using the full training split and no augmentation.

The goal is a reference result on the full training set. Later experiments (augmentation, class-imbalance handling, fine-tuning) should be compared against it.


## Dataset

The split file contains 53,705 image records across 38 classes.

- Training: 37,751 images
- Validation: 7,976 images
- Test: 7,978 images

The original PlantVillage images are not included in this repository. Attach the PlantVillage dataset containing the colour images as an input in Kaggle. Also attach `split.csv` as an input. Its columns are `path`, `class` and `split`, and the paths are relative to the `color` folder.

The script uses the existing split. Do not create another random split when running this experiment.


## Model and training

| Setting | Value |
| --- | --- |
| Backbone | ResNet-50, torchvision `IMAGENET1K_V2` weights, all layers frozen and kept in eval mode |
| Head | Dropout 0.2, then Linear (2048 to 38) |
| Input | Resize to 224x224, ImageNet mean and std normalisation |
| Augmentation | None |
| Loss | Cross-entropy, no class weights |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule |
| Training | 30 epochs, head batch size 256, seed 42 |
| Model selection | Best validation macro-F1 (epoch 28) |

Because the backbone is frozen and there is no augmentation, each image always produces the same 2048-number feature vector. The script extracts the features once for train, validation and test, then trains the head on the cached features. This is equivalent to training the head on top of a frozen ResNet-50. Feature extraction took about 31 minutes on a Kaggle T4 GPU, and training the head is fast.

The evaluation transform (`build_eval_transform` in `augmentation_pipeline.py`) only resizes and normalises the image. The test split was evaluated once, after the epoch was selected on validation.


## How to run

Use a Kaggle notebook with GPU and Internet on (Internet is needed to download the ImageNet weights). Put `train_baseline.py`, `augmentation_pipeline.py` and `split.csv` in the working folder, using exactly these names, then run:

```bash
python train_baseline.py \
  --csv split.csv \
  --root /kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color \
  --out results
```

`--root` is the folder that contains the 38 class folders. The script checks that every image path exists before starting. Requirements: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`, `opencv-python` (OpenCV is imported by `augmentation_pipeline.py`). The run used PyTorch 2.11.0 and OpenCV 4.14.0.


## Results

| Split | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Validation | 97.40% | 0.9689 | 0.9739 |
| Test | 97.00% | 0.9635 | 0.9697 |

Weakest classes by F1:

| Split | Lowest-F1 classes |
| --- | --- |
| Validation | Corn Cercospora/Gray leaf spot 0.833, Tomato Early blight 0.879, Tomato Target Spot 0.913 |
| Test | Tomato Early blight 0.824, Potato healthy 0.850 (23 test images), Corn Cercospora/Gray leaf spot 0.857, Tomato Target Spot 0.880 |

Validation macro-F1 rose from 0.810 at epoch 1 to about 0.969 and changed by less than 0.002 over the last ten epochs, so training had converged.

This is the full-training-set, no-augmentation baseline. Its validation macro-F1 (0.9689) is much higher than the 1,900-image screening baseline (0.8051), because the head here is trained on all 37,751 training images.


## Output files

Everything is written to the `--out` folder (`results/`):

| File | Contents |
| --- | --- |
| `metrics.json` | Validation and test accuracy, macro-F1, weighted-F1, best epoch, settings |
| `val_per_class.csv`, `test_per_class.csv` | Precision, recall, F1 and support for each class |
| `val_report.txt`, `test_report.txt` | The same per-class numbers as text |
| `val_confusion_matrix.csv/.png`, `test_confusion_matrix.csv/.png` | Raw counts and a row-normalised plot |
| `training_curve.csv/.png` | Train loss and validation accuracy / macro-F1 per epoch |
| `train_class_counts.csv` | Training images per class |
| `classes.txt`, `head_best.pt` | Class names and the trained head weights |

Cached features and per-image predictions are not saved.


## Current limitations and next steps

- This is a single run with one seed, so there are no confidence intervals. Small differences between configurations should not be over-interpreted.
- The hyperparameters are standard defaults and were not tuned.
- PlantVillage images are lab-controlled, so these scores likely overstate performance on field images.
- Near-duplicate removal can miss some copies, and photos of the same leaf may fall in different splits. This could inflate validation and test scores. A near-duplicate check between train and test should be reported.
- Cross-class near-duplicate matches were kept in the split, so a few images may carry conflicting labels.
- A frozen backbone cannot adapt to leaf-specific patterns, which probably limits similar-looking classes such as the tomato diseases.
- Small classes give noisy scores. Potato healthy has only 23 test images.
- A full-training-set augmented run for comparison was started but not completed, so the full-data comparison with augmentation is still needed. Use validation macro-F1 for that comparison and keep the test split for the final configuration.
- Still to do: class-weighted loss or sampling, fine-tuning the backbone, calibration, and Grad-CAM.
