"""Configuration settings for ViT-Tiny training and evaluation on PlantVillage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_CSV = BASE_DIR / "split.csv"
DEFAULT_CLASS_TO_IDX = BASE_DIR / "class_to_idx.json"
DEFAULT_TOP5_CONFIGS_JSON = BASE_DIR / "vit_tiny_top5_configs.json"

# Dataset Specification
NUM_CLASSES = 38
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Expected dataset split counts
EXPECTED_TRAIN_ROWS = 37751
EXPECTED_VAL_ROWS = 7976
EXPECTED_TEST_ROWS = 7978
EXPECTED_TOTAL_ROWS = 53705

# Model Architecture
MODEL_NAME = "vit_tiny_patch16_224"
PRETRAINED = True
PRETRAINED_WEIGHTS = "ImageNet-1K"
TRAINABLE_SCOPE = "full_fine_tuning"

# Optimization Hyperparameters
OPTIMIZER_NAME = "AdamW"
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 5e-4
WEIGHT_DECAY = 1e-4
LOSS_FUNCTION = "CrossEntropyLoss"
LR_SCHEDULER = None

# Training & DataLoader parameters
EPOCHS = 3
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
SEEDS = (42, 123, 2026)
MIXED_PRECISION = True  # FP16 autocast + GradScaler
NUM_WORKERS = 4
PREFETCH_FACTOR = 4

# Augmentation Recipes (authoritative from terravista_all_logged_configs_ranked.json)
AUGMENTATION_RECIPES: dict[str, dict[str, float]] = {
    "00_no_augmentation": {
        "p_flip": 0.0,
        "p_rotation": 0.0,
        "p_colour": 0.0,
        "p_blur": 0.0,
        "p_noise": 0.0,
        "p_cutout": 0.0,
        "p_background": 0.0,
    },
    "03_uniform_p010": {
        "p_flip": 0.10,
        "p_rotation": 0.10,
        "p_colour": 0.10,
        "p_blur": 0.10,
        "p_noise": 0.10,
        "p_cutout": 0.10,
        "p_background": 0.10,
    },
    "02_uniform_p020": {
        "p_flip": 0.20,
        "p_rotation": 0.20,
        "p_colour": 0.20,
        "p_blur": 0.20,
        "p_noise": 0.20,
        "p_cutout": 0.20,
        "p_background": 0.20,
    },
    "01_joint_refined_recipe": {
        "p_flip": 0.0,
        "p_rotation": 0.0,
        "p_colour": 0.25,
        "p_blur": 0.10,
        "p_noise": 0.25,
        "p_cutout": 0.25,
        "p_background": 0.10,
    },
    "04_uniform_p025": {
        "p_flip": 0.25,
        "p_rotation": 0.25,
        "p_colour": 0.25,
        "p_blur": 0.25,
        "p_noise": 0.25,
        "p_cutout": 0.25,
        "p_background": 0.25,
    },
}

RECIPE_ALIASES = {
    "no_aug": "00_no_augmentation",
    "no_augmentation": "00_no_augmentation",
    "uniform_p010": "03_uniform_p010",
    "uniform_p020": "02_uniform_p020",
    "joint_refined": "01_joint_refined_recipe",
    "uniform_p025": "04_uniform_p025",
}


def resolve_recipe_name(name: str) -> str:
    """Normalize recipe name to canonical name."""
    clean = name.strip()
    return RECIPE_ALIASES.get(clean, clean)


def get_recipe_probabilities(name: str) -> dict[str, float]:
    """Retrieve augmentation probability vector for a recipe."""
    canonical = resolve_recipe_name(name)
    if canonical not in AUGMENTATION_RECIPES:
        raise ValueError(
            f"Unknown recipe '{name}'. Available: {list(AUGMENTATION_RECIPES.keys())} or {list(RECIPE_ALIASES.keys())}"
        )
    return dict(AUGMENTATION_RECIPES[canonical])


def to_dict() -> dict[str, Any]:
    """Export complete configuration dictionary for logging and handoff."""
    return {
        "dataset": {
            "name": "PlantVillage",
            "num_classes": NUM_CLASSES,
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "image_channels": IMAGE_CHANNELS,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "expected_counts": {
                "train": EXPECTED_TRAIN_ROWS,
                "val": EXPECTED_VAL_ROWS,
                "test": EXPECTED_TEST_ROWS,
                "total": EXPECTED_TOTAL_ROWS,
            },
        },
        "model": {
            "name": MODEL_NAME,
            "library": "timm",
            "pretrained": PRETRAINED,
            "pretrained_weights": PRETRAINED_WEIGHTS,
            "trainable_scope": TRAINABLE_SCOPE,
            "num_classes": NUM_CLASSES,
        },
        "optimization": {
            "optimizer": OPTIMIZER_NAME,
            "backbone_lr": BACKBONE_LR,
            "classifier_lr": CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY,
            "loss": LOSS_FUNCTION,
            "lr_scheduler": LR_SCHEDULER,
        },
        "training": {
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "seeds": list(SEEDS),
            "mixed_precision": MIXED_PRECISION,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
        },
        "augmentation_recipes": AUGMENTATION_RECIPES,
    }
