# Plant Disease Image Augmentation

## Overview

This project prepares and tests image augmentation methods for the TerraVista PlantVillage crop disease classification task. It uses PyTorch and torchvision.

The goal is to create realistic variations of training images and check whether they improve validation performance. The notebook also compares the augmented runs with a no-augmentation baseline.


## Dataset

The split file contains 53,705 image records across 38 classes.

- Training: 37,751 images
- Validation: 7,976 images
- Test: 7,978 images

The original PlantVillage images are not included in this repository. Attach the PlantVillage dataset containing the colour images as an input in Kaggle. Also attach `data/split.csv` as an input.

The notebook uses the existing split. Do not create another random split when running this experiment.

## Augmentation methods

The training transform can apply:

- Horizontal and vertical flips
- Rotation and zoom
- Brightness, contrast, saturation, and hue changes
- Gaussian blur
- Gaussian sensor noise
- Cutout, with a maximum area of 20 percent
- Optional background replacement using an estimated foreground mask

The evaluation transform only resizes the image and converts it to a tensor. It does not apply random augmentation. Augmentation should be used only for training images.

Background replacement uses an estimated leaf mask. Check the saved mask and background previews carefully because an incorrect mask can remove part of a leaf or a disease lesion.

## Tuning method

The notebook uses a pretrained ResNet-50 as a frozen feature extractor and trains a new linear classification head for 38 classes. It screens augmentation configurations using up to 50 training images per class, for 1,900 training images in total. It evaluates each candidate on the full validation split of 7,976 images.

Configurations are selected using validation macro-F1, with validation accuracy also reported. The test split is not used for tuning.

## Screening results

| Configuration | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: |
| No augmentation baseline | 84.00% | 0.8051 |
| Cutout 10% and moderate zoom | 80.52% | 0.7681 |
| Cutout 15% and moderate zoom | 81.11% | 0.7767 |
| Cutout 20% and moderate zoom | 81.26% | 0.7740 |
| Cutout 15% and background replacement at 10% probability | 80.04% | 0.7614 |
| Wider zoom and rotation, cutout 15%, and background replacement at 10% probability | 78.60% | 0.7478 |

The no-augmentation baseline performed best in this screening run. The best augmented candidate was cutout 15% with moderate zoom and no background replacement, but it did not outperform the baseline.

These are screening results, not final full-training-set model results. The experiment uses a small training subset and a frozen feature extractor. Run a full-training-set comparison before making final model performance claims. The test set must remain unused until the final configuration is selected.


## Current limitations and next steps

- The augmentation settings have only been screened on a subset of the training data.
- The full-training-set baseline and augmented comparison are still needed.
- The final project repository should include the script that recreates `split.csv` from the cleaned dataset using a fixed seed and stratified splitting.
- Duplicate and near-duplicate checks across training, validation, and test should be reported separately.
- Keep validation and test images unaugmented.
