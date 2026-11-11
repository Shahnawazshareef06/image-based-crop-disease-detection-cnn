"""Dataset loading, validation, and PyTorch DataLoader integration for PlantVillage."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import (
    DEFAULT_CLASS_TO_IDX,
    DEFAULT_SPLIT_CSV,
    EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    IMAGE_SIZE,
    NUM_CLASSES,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
)


def normalize_relative_path(path_str: str) -> str:
    """Standardize path separators and strip leading redundant 'color/' prefixes."""
    clean = str(path_str).replace("\\", "/").strip().lstrip("/")
    if clean.lower().startswith("color/"):
        clean = clean[6:]
    return clean


def verify_dataset_manifest(
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    class_to_idx_path: str | Path = DEFAULT_CLASS_TO_IDX,
    image_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Audit split manifest against project requirements and integrity criteria.

    Checks:
    1. Row counts per split (train: 37,751, val: 7,976, test: 7,978, total: 53,705).
    2. Representation of all 38 classes in every split.
    3. Duplicate image paths.
    4. Cross-split data leakage.
    5. Folder name vs. class label consistency.
    6. Presence of missing/corrupted image files on disk (if image_root is provided).
    """
    split_csv = Path(split_csv)
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found at: {split_csv}")

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx: dict[str, int] = json.load(f)

    if len(class_to_idx) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes in class_to_idx, found {len(class_to_idx)}")

    split_counts: Counter[str] = Counter()
    class_split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    all_paths: set[str] = set()
    split_paths: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    duplicate_paths: list[str] = []
    folder_mismatches: list[tuple[str, str, str]] = []

    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"path", "class", "split"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Split CSV must contain columns {required_cols}; found {reader.fieldnames}")

        for idx, row in enumerate(reader):
            raw_path = row["path"]
            cls_name = row["class"]
            split = row["split"].strip().lower()
            if split in ("validation", "valid"):
                split = "val"

            split_counts[split] += 1
            if split in class_split_counts:
                class_split_counts[split][cls_name] += 1

            rel_path = normalize_relative_path(raw_path)
            if rel_path in all_paths:
                duplicate_paths.append(rel_path)
            all_paths.add(rel_path)
            if split in split_paths:
                split_paths[split].add(rel_path)

            folder = rel_path.split("/")[0] if "/" in rel_path else ""
            if folder and folder != cls_name:
                folder_mismatches.append((rel_path, folder, cls_name))

    # Cross-split leakage checks
    train_val_leakage = split_paths["train"].intersection(split_paths["val"])
    train_test_leakage = split_paths["train"].intersection(split_paths["test"])
    val_test_leakage = split_paths["val"].intersection(split_paths["test"])
    total_leakage = len(train_val_leakage) + len(train_test_leakage) + len(val_test_leakage)

    # Class coverage
    expected_classes = set(class_to_idx.keys())
    missing_classes_by_split = {
        s: list(expected_classes - set(counts.keys()))
        for s, counts in class_split_counts.items()
    }

    # Disk file check (if image_root provided)
    missing_on_disk = 0
    corrupted_files = 0
    disk_checked = False
    if image_root is not None:
        root_path = Path(image_root)
        if root_path.is_dir():
            disk_checked = True
            for p in all_paths:
                full_path = root_path / p
                if not full_path.is_file():
                    missing_on_disk += 1
                else:
                    try:
                        with Image.open(full_path) as img:
                            img.verify()
                    except Exception:
                        corrupted_files += 1

    report = {
        "split_csv_path": str(split_csv),
        "total_rows": sum(split_counts.values()),
        "split_counts": dict(split_counts),
        "counts_match_expected": {
            "train": split_counts["train"] == EXPECTED_TRAIN_ROWS,
            "val": split_counts["val"] == EXPECTED_VAL_ROWS,
            "test": split_counts["test"] == EXPECTED_TEST_ROWS,
            "total": sum(split_counts.values()) == EXPECTED_TOTAL_ROWS,
        },
        "unique_paths_count": len(all_paths),
        "duplicates_count": len(duplicate_paths),
        "leakage": {
            "train_val_overlap": len(train_val_leakage),
            "train_test_overlap": len(train_test_leakage),
            "val_test_overlap": len(val_test_leakage),
            "total_leakage": total_leakage,
        },
        "folder_class_mismatches": len(folder_mismatches),
        "missing_classes_by_split": missing_classes_by_split,
        "all_38_classes_represented": all(
            len(missing) == 0 for missing in missing_classes_by_split.values()
        ),
        "disk_verification": {
            "performed": disk_checked,
            "image_root": str(image_root) if image_root else None,
            "missing_on_disk": missing_on_disk,
            "corrupted_files": corrupted_files,
        },
    }
    return report


def load_split_frames(
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    class_to_idx: Optional[Mapping[str, int]] = None,
    allow_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Read split.csv into structured train, val, and (optional) test DataFrames.

    Test split is only returned when allow_test=True, strictly enforcing the test set restriction.
    """
    split_csv = Path(split_csv)
    df = pd.read_csv(split_csv)
    df["split"] = (
        df["split"].astype(str).str.strip().str.lower()
        .replace({"validation": "val", "valid": "val"})
    )
    df["relative_path"] = df["path"].map(normalize_relative_path)

    if class_to_idx is not None:
        df["target"] = df["class"].map(class_to_idx)
        if df["target"].isna().any():
            unmapped = df[df["target"].isna()]["class"].unique()
            raise ValueError(f"Unmapped classes found: {unmapped}")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True) if allow_test else None

    return train_df, val_df, test_df


class PlantVillageDataset(Dataset):
    """PyTorch Dataset yielding uint8 RGB tensors [224, 224, 3] and target class IDs."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Optional[str | Path] = None,
        image_size: int = IMAGE_SIZE,
        need_background_masks: bool = False,
        mask_cache: Optional[np.ndarray] = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.need_background_masks = need_background_masks
        self.mask_cache = mask_cache

        self.paths = self.frame["relative_path"].tolist()
        self.targets = self.frame["target"].tolist() if "target" in self.frame else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, torch.Tensor]:
        rel_path = self.paths[index]
        label = self.targets[index] if self.targets is not None else -1

        if self.image_root is not None and (self.image_root / rel_path).is_file():
            img_path = self.image_root / rel_path
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                image_arr = np.asarray(img_rgb, dtype=np.uint8)
        else:
            # Deterministic synthetic placeholder for dry-run/pipeline verification
            image_arr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image_tensor = torch.from_numpy(image_arr)

        if self.need_background_masks:
            if self.mask_cache is not None and index < len(self.mask_cache):
                mask_arr = self.mask_cache[index]
            else:
                # Default foreground mask (all 255: leaf preserved)
                mask_arr = np.full((112, 112), 255, dtype=np.uint8)
            mask_tensor = torch.from_numpy(mask_arr)
            return image_tensor, label, mask_tensor

        return image_tensor, label


def make_reproducible_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Generate worker_init_fn to ensure deterministic DataLoader workers."""
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id * 1000
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_root: Optional[str | Path],
    seed: int,
    need_background_masks: bool = False,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    val_batch_size: int = VAL_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    prefetch_factor: int = PREFETCH_FACTOR,
) -> tuple[DataLoader, DataLoader]:
    """Construct deterministic PyTorch DataLoaders for training and validation."""
    train_dataset = PlantVillageDataset(
        frame=train_df,
        image_root=image_root,
        need_background_masks=need_background_masks,
    )
    val_dataset = PlantVillageDataset(
        frame=val_df,
        image_root=image_root,
        need_background_masks=False,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    kwargs = {
        "num_workers": num_workers,
        "worker_init_fn": make_reproducible_worker_init_fn(seed),
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=g,
        **kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, val_loader
