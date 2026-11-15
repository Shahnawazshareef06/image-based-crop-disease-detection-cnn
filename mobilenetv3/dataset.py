"""PlantVillage train/validation data adapter with notebook-compatible caches.

- Only train and validation manifest rows are retained and resolved.
- RGB images are resized with PIL bilinear to 224x224 and stored in an mmap cache.
- GrabCut alpha masks are generated from training images only and stored in a
  separate mmap cache indexed by the ordered train frame.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from augmentation_pipeline import (
    BACKGROUND_MASK_SIZE,
    BACKGROUND_MASK_CACHE_VERSION,
    estimate_foreground_alpha_from_rgb,
)
from config import IMAGE_SIZE

CACHE_WORKERS = 4


def normalize_relative_path(value: str) -> str:
    value = str(value).replace("\\", "/").strip().lstrip("/")
    if value.lower().startswith("color/"):
        value = value[6:]
    return value


def load_train_val_frames(
    split_csv: str | Path,
    class_to_idx: Mapping[str, int],
    *,
    expected_train_rows: int | None = 37751,
    expected_validation_rows: int | None = 7976,
):
    """Load train/val rows only, preserving their original relative order."""
    frame = pd.read_csv(split_csv)
    required = {"path", "class", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Split manifest must contain {sorted(required)}; found {list(frame.columns)}")
    frame["split"] = (
        frame["split"].astype(str).str.strip().str.lower()
        .replace({"validation": "val", "valid": "val"})
    )
    # Test and any other split labels are discarded before image paths are resolved.
    train_df = frame.loc[frame["split"] == "train", ["path", "class", "split"]].copy().reset_index(drop=True)
    val_df = frame.loc[frame["split"] == "val", ["path", "class", "split"]].copy().reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(f"Expected non-empty train/val rows; found train={len(train_df)}, val={len(val_df)}")
    if expected_train_rows is not None and len(train_df) != expected_train_rows:
        raise ValueError(f"Expected {expected_train_rows} train rows, got {len(train_df)}")
    if expected_validation_rows is not None and len(val_df) != expected_validation_rows:
        raise ValueError(f"Expected {expected_validation_rows} validation rows, got {len(val_df)}")

    train_classes = sorted(train_df["class"].astype(str).unique().tolist())
    val_classes = sorted(val_df["class"].astype(str).unique().tolist())
    if train_classes != val_classes or set(train_classes) != set(class_to_idx):
        raise ValueError("Train/validation classes do not match the bundled class_to_idx.json.")
    if set(class_to_idx.values()) != set(range(len(class_to_idx))):
        raise ValueError("class_to_idx values must be contiguous integers starting at zero.")

    for split_name, split_frame in (("train", train_df), ("val", val_df)):
        split_frame["relative_path"] = split_frame["path"].map(normalize_relative_path)
        folder_labels = split_frame["relative_path"].str.split("/").str[0]
        if not (folder_labels == split_frame["class"].astype(str)).all():
            raise ValueError(f"Folder/class mismatch in {split_name} split.")
    all_used_paths = pd.concat([train_df["relative_path"], val_df["relative_path"]], ignore_index=True)
    if all_used_paths.duplicated().any():
        raise ValueError("Duplicate image path found across train/validation rows.")
    return train_df, val_df


def notebook_cache_fingerprint(train_df: pd.DataFrame, val_df: pd.DataFrame) -> str:
    """Recreate the original notebook's SHA-256 fingerprint for train+val rows."""
    ordered = pd.concat([train_df, val_df], ignore_index=True)
    digest_source = (
        ordered["split"].astype(str) + "\t" + ordered["relative_path"].astype(str)
        + "\t" + ordered["class"].astype(str)
    ).str.cat(sep="\n")
    return hashlib.sha256(f"{IMAGE_SIZE}\n{digest_source}".encode("utf-8")).hexdigest()


def build_decoded_image_cache(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    color_dir: str | Path,
    cache_dir: str | Path,
    *,
    workers: int = CACHE_WORKERS,
    reuse_cache_dirs=(),
) -> tuple[Path, str]:
    """Build/reuse/resume the same ordered train+val 224px uint8 RGB cache.

    `reuse_cache_dirs` may contain read-only Kaggle attached-output cache folders.
    Matching cache files are opened in place; they are not modified.
    """
    color_dir = Path(color_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = notebook_cache_fingerprint(train_df, val_df)
    final_path = cache_dir / f"plantvillage_train_val_{IMAGE_SIZE}_{fingerprint[:12]}.npy"
    meta_path = final_path.with_suffix(".json")
    partial_path = final_path.with_name(final_path.stem + ".partial.npy")
    progress_path = final_path.with_name(final_path.stem + ".progress.npy")

    ordered = pd.concat([train_df, val_df], ignore_index=True)
    paths = [color_dir / p for p in ordered["relative_path"].tolist()]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} train/validation images; first missing path: {missing[0]}")
    shape = (len(ordered), IMAGE_SIZE, IMAGE_SIZE, 3)

    # Search attached prior-version outputs before creating a new local cache.
    for source_dir in reuse_cache_dirs or ():
        candidate = Path(source_dir) / final_path.name
        if candidate.is_file():
            try:
                source_cache = np.load(candidate, mmap_mode="r")
                source_valid = tuple(source_cache.shape) == shape and source_cache.dtype == np.uint8
                del source_cache
                if source_valid:
                    print("Reusing attached prior-version image cache read-only:", candidate)
                    return candidate, fingerprint
            except Exception:
                pass

    if final_path.is_file():
        try:
            cache = np.load(final_path, mmap_mode="r")
            shape_ok = tuple(cache.shape) == shape and cache.dtype == np.uint8
            del cache
            if shape_ok:
                return final_path, fingerprint
        except Exception:
            pass
        final_path.unlink(missing_ok=True)

    if partial_path.is_file() and progress_path.is_file():
        cache = np.load(partial_path, mmap_mode="r+")
        progress = np.load(progress_path, mmap_mode="r+")
        if tuple(cache.shape) != shape or cache.dtype != np.uint8 or tuple(progress.shape) != (len(ordered),):
            del cache, progress
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            cache = progress = None
    else:
        cache = progress = None

    if cache is None:
        required_bytes = int(np.prod(shape)) + 128 * 1024 * 1024
        if __import__("shutil").disk_usage(cache_dir).free < required_bytes:
            raise OSError(f"Insufficient free space for decoded image cache (~{np.prod(shape)/(1024**3):.2f} GiB).")
        cache = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.uint8, shape=shape)
        progress = np.lib.format.open_memmap(progress_path, mode="w+", dtype=np.uint8, shape=(len(ordered),))
        cache[:] = 0
        progress[:] = 0
        cache.flush(); progress.flush()

    done_before = int(np.asarray(progress).sum())
    remaining = np.flatnonzero(np.asarray(progress) == 0)

    def decode_one(index):
        with Image.open(paths[int(index)]) as image:
            rgb = np.asarray(
                image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        return int(index), rgb

    print(f"Decoded image cache: resuming {done_before:,}/{len(ordered):,} rows; workers={workers}")
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for start in range(0, len(remaining), 128):
                chunk = remaining[start:start + 128]
                for index, rgb in pool.map(decode_one, chunk):
                    cache[index] = rgb
                    progress[index] = 1
                cache.flush(); progress.flush()
    except Exception:
        cache.flush(); progress.flush()
        del cache, progress
        raise

    completed = int(np.asarray(progress).sum())
    cache.flush(); progress.flush()
    del cache, progress
    if completed != len(ordered):
        raise RuntimeError(f"Decoded image cache incomplete ({completed}/{len(ordered)}); rerun to resume.")

    os.replace(partial_path, final_path)
    progress_path.unlink(missing_ok=True)
    meta_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "shape": shape,
        "dtype": "uint8",
        "image_size": IMAGE_SIZE,
        "rows": len(ordered),
        "source": "train + validation rows only",
    }, indent=2), encoding="utf-8")
    return final_path, fingerprint


def build_training_alpha_cache(
    train_df: pd.DataFrame,
    image_cache_path: str | Path,
    cache_dir: str | Path,
    cache_fingerprint: str,
    *,
    workers: int = CACHE_WORKERS,
    reuse_cache_dirs=(),
) -> Path:
    """Build/reuse/resume GrabCut masks for training rows only, in exact train order.

    `reuse_cache_dirs` may contain read-only Kaggle attached-output cache folders.
    Matching cache files are opened in place; they are not modified.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask_fingerprint = hashlib.sha256(
        f"{cache_fingerprint}\n{BACKGROUND_MASK_CACHE_VERSION}\n{BACKGROUND_MASK_SIZE}".encode("utf-8")
    ).hexdigest()
    final_path = cache_dir / f"foreground_alpha_train_{BACKGROUND_MASK_SIZE}_{mask_fingerprint[:12]}.npy"
    meta_path = final_path.with_suffix(".json")
    partial_path = final_path.with_name(final_path.stem + ".partial.npy")
    progress_path = final_path.with_name(final_path.stem + ".progress.npy")
    shape = (len(train_df), BACKGROUND_MASK_SIZE, BACKGROUND_MASK_SIZE)

    for source_dir in reuse_cache_dirs or ():
        candidate = Path(source_dir) / final_path.name
        if candidate.is_file():
            try:
                source_masks = np.load(candidate, mmap_mode="r")
                source_valid = tuple(source_masks.shape) == shape and source_masks.dtype == np.uint8
                del source_masks
                if source_valid:
                    print("Reusing attached prior-version training mask cache read-only:", candidate)
                    return candidate
            except Exception:
                pass

    if final_path.is_file():
        try:
            existing = np.load(final_path, mmap_mode="r")
            shape_ok = tuple(existing.shape) == shape and existing.dtype == np.uint8
            del existing
            if shape_ok:
                return final_path
        except Exception:
            pass
        final_path.unlink(missing_ok=True)

    if partial_path.is_file() and progress_path.is_file():
        masks = np.load(partial_path, mmap_mode="r+")
        progress = np.load(progress_path, mmap_mode="r+")
        if tuple(masks.shape) != shape or masks.dtype != np.uint8 or tuple(progress.shape) != (len(train_df),):
            del masks, progress
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            masks = progress = None
    else:
        masks = progress = None

    if masks is None:
        required_bytes = int(np.prod(shape)) + 64 * 1024 * 1024
        if __import__("shutil").disk_usage(cache_dir).free < required_bytes:
            raise OSError(f"Insufficient free space for training alpha cache (~{np.prod(shape)/(1024**3):.2f} GiB).")
        masks = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.uint8, shape=shape)
        progress = np.lib.format.open_memmap(progress_path, mode="w+", dtype=np.uint8, shape=(len(train_df),))
        masks[:] = 255
        progress[:] = 0
        masks.flush(); progress.flush()

    image_cache = np.load(image_cache_path, mmap_mode="r")
    done_before = int(np.asarray(progress).sum())
    remaining = np.flatnonzero(np.asarray(progress) == 0)

    def make_one(index):
        alpha, _valid = estimate_foreground_alpha_from_rgb(
            np.asarray(image_cache[int(index)]), output_size=BACKGROUND_MASK_SIZE
        )
        return int(index), alpha

    print(f"Training-only alpha masks: resuming {done_before:,}/{len(train_df):,} rows; workers={workers}")
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for start in range(0, len(remaining), 64):
                chunk = remaining[start:start + 64]
                for index, alpha in pool.map(make_one, chunk):
                    masks[index] = alpha
                    progress[index] = 1
                masks.flush(); progress.flush()
    except Exception:
        masks.flush(); progress.flush()
        del masks, progress, image_cache
        raise

    completed = int(np.asarray(progress).sum())
    masks.flush(); progress.flush()
    del masks, progress, image_cache
    if completed != len(train_df):
        raise RuntimeError(f"Training alpha cache incomplete ({completed}/{len(train_df)}); rerun to resume.")

    os.replace(partial_path, final_path)
    progress_path.unlink(missing_ok=True)
    meta_path.write_text(json.dumps({
        "fingerprint": mask_fingerprint,
        "shape": shape,
        "dtype": "uint8",
        "mask_size": BACKGROUND_MASK_SIZE,
        "rows": len(train_df),
        "algorithm": BACKGROUND_MASK_CACHE_VERSION,
        "source": "training images only",
    }, indent=2), encoding="utf-8")
    return final_path


class CachedPlantVillageDataset(Dataset):
    """Lazy mmap Dataset yielding RGB uint8 NHWC samples in original manifest order."""
    def __init__(self, positions, label_ids, image_cache_path, *, alpha_cache_path=None):
        self.positions = np.asarray(positions, dtype=np.int64)
        self.label_ids = np.asarray(label_ids, dtype=np.int64)
        self.image_cache_path = str(image_cache_path)
        self.alpha_cache_path = str(alpha_cache_path) if alpha_cache_path else None
        self._images = None
        self._masks = None

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        if self._images is None:
            self._images = np.load(self.image_cache_path, mmap_mode="r")
        cache_pos = int(self.positions[index])
        image = np.array(self._images[cache_pos], copy=True)
        label = int(self.label_ids[index])
        if self.alpha_cache_path is None:
            return torch.from_numpy(image), label
        if self._masks is None:
            self._masks = np.load(self.alpha_cache_path, mmap_mode="r")
        alpha = np.array(self._masks[index], copy=True)
        return torch.from_numpy(image), label, torch.from_numpy(alpha)


def make_plantvillage_datasets(
    split_csv,
    color_dir,
    class_to_idx,
    *,
    need_background_masks,
    cache_dir,
    reuse_cache_dirs=(),
):
    train_df, val_df = load_train_val_frames(split_csv, class_to_idx)
    image_cache_path, fingerprint = build_decoded_image_cache(
        train_df, val_df, color_dir, cache_dir, reuse_cache_dirs=reuse_cache_dirs
    )
    mask_path = None
    if need_background_masks:
        mask_path = build_training_alpha_cache(
            train_df, image_cache_path, cache_dir, fingerprint, reuse_cache_dirs=reuse_cache_dirs
        )

    train_labels = train_df["class"].astype(str).map(class_to_idx).to_numpy(dtype=np.int64)
    val_labels = val_df["class"].astype(str).map(class_to_idx).to_numpy(dtype=np.int64)
    train_positions = np.arange(len(train_df), dtype=np.int64)
    val_positions = np.arange(len(train_df), len(train_df) + len(val_df), dtype=np.int64)
    train_dataset = CachedPlantVillageDataset(
        train_positions, train_labels, image_cache_path, alpha_cache_path=mask_path
    )
    val_dataset = CachedPlantVillageDataset(
        val_positions, val_labels, image_cache_path, alpha_cache_path=None
    )
    return train_dataset, val_dataset, train_df, val_df, mask_path, image_cache_path, fingerprint
