"""ViT-Tiny training pipeline with reproducible multi-seed augmentation evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import (
    AUGMENTATION_RECIPES,
    BACKBONE_LR,
    CLASSIFIER_LR,
    DEFAULT_CLASS_TO_IDX,
    DEFAULT_SPLIT_CSV,
    EPOCHS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LOSS_FUNCTION,
    MIXED_PRECISION,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_WORKERS,
    OPTIMIZER_NAME,
    PREFETCH_FACTOR,
    RECIPE_ALIASES,
    SEEDS,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    WEIGHT_DECAY,
    get_recipe_probabilities,
    resolve_recipe_name,
)
from dataset import (
    PlantVillageDataset,
    build_dataloaders,
    load_split_frames,
    verify_dataset_manifest,
)
from evaluate import (
    compute_metrics,
    evaluate_model,
    find_most_confused_pairs,
    find_weakest_classes,
    generate_classification_report,
    generate_predictions_dataframe,
    plot_confusion_matrix,
)
from model import (
    benchmark_inference,
    build_vit_tiny,
    count_parameters,
    get_optimizer_param_groups,
)

# Import existing team augmentation pipeline
try:
    from augmentation_pipeline import (
        PROBABILITY_KEYS,
        augment_batch_gpu,
        validate_probability_config,
    )
except ImportError:
    from TerraVista_Reproducible_Team_Handoff.augmentation_pipeline import (
        PROBABILITY_KEYS,
        augment_batch_gpu,
        validate_probability_config,
    )


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[GradScaler],
    device: torch.device,
    aug_config: Mapping[str, float],
    use_amp: bool = MIXED_PRECISION,
) -> float:
    """Train model for one epoch using FP16 mixed precision and GPU augmentation."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images_uint8 = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        masks = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            # Augment batch on device returning normalized float NCHW tensors
            aug_images = augment_batch_gpu(
                images_uint8, aug_config, device=device, background_alpha_uint8=masks
            )
            outputs = model(aug_images)
            loss = criterion(outputs, targets)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def train_one_recipe_seed(
    recipe_name: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    idx_to_class: Mapping[int, str],
    image_root: Optional[str | Path],
    output_dir: Path,
    epochs: int = EPOCHS,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    val_batch_size: int = VAL_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    prefetch_factor: int = PREFETCH_FACTOR,
    use_amp: bool = MIXED_PRECISION,
    pretrained: bool = True,
) -> dict[str, Any]:
    """Execute training and validation for a single (recipe, seed) run."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Run: Recipe '{recipe_name}' | Seed {seed} | Device: {device}")
    print(f"=======================================================")

    aug_config = get_recipe_probabilities(recipe_name)
    validate_probability_config(aug_config)
    need_masks = aug_config.get("p_background", 0.0) > 0.0

    train_loader, val_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        image_root=image_root,
        seed=seed,
        need_background_masks=need_masks,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    model = build_vit_tiny(num_classes=NUM_CLASSES, pretrained=pretrained)
    model.to(device)

    param_groups = get_optimizer_param_groups(
        model,
        backbone_lr=BACKBONE_LR,
        classifier_lr=CLASSIFIER_LR,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device=device.type) if (use_amp and device.type == "cuda") else None

    run_dir = output_dir / f"{recipe_name}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / f"best_vit_tiny_{recipe_name}_seed_{seed}.pth"

    best_macro_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    best_epoch = -1
    best_predictions: tuple[list[int], list[int]] = ([], [])
    epoch_history: list[dict[str, Any]] = []

    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            aug_config=aug_config,
            use_amp=use_amp,
        )

        val_loss, y_true, y_pred = evaluate_model(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, loss=val_loss, train_loss=train_loss)
        epoch_time = time.perf_counter() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_macro_precision": metrics["macro_precision"],
            "val_macro_recall": metrics["macro_recall"],
            "epoch_duration_seconds": round(epoch_time, 2),
        }
        epoch_history.append(epoch_record)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_metrics = dict(metrics)
            best_epoch = epoch
            best_predictions = (y_true, y_pred)

            torch.save(
                {
                    "epoch": epoch,
                    "recipe_name": recipe_name,
                    "seed": seed,
                    "model_name": MODEL_NAME,
                    "num_classes": NUM_CLASSES,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": best_macro_f1,
                    "metrics": best_metrics,
                    "aug_config": aug_config,
                },
                best_checkpoint_path,
            )

    total_duration = time.perf_counter() - start_train_time

    # Save epoch history to CSV
    pd.DataFrame(epoch_history).to_csv(run_dir / "epoch_history.csv", index=False)

    run_summary = {
        "recipe_name": recipe_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "best_metrics": best_metrics,
        "total_training_duration_seconds": round(total_duration, 2),
        "best_checkpoint_path": str(best_checkpoint_path),
        "probabilities": aug_config,
        "epoch_history": epoch_history,
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    return {
        "recipe_name": recipe_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "metrics": best_metrics,
        "checkpoint_path": best_checkpoint_path,
        "y_true": best_predictions[0],
        "y_pred": best_predictions[1],
        "summary": run_summary,
    }


def run_full_study(
    recipes: Sequence[str],
    seeds: Sequence[int] = SEEDS,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    class_to_idx_path: str | Path = DEFAULT_CLASS_TO_IDX,
    image_root: Optional[str | Path] = None,
    output_dir: str | Path = "./vit_tiny_runs",
    epochs: int = EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict[str, Any]:
    """Execute all selected recipes across all seeds and aggregate comparison results."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Verify dataset split manifest
    manifest_audit = verify_dataset_manifest(split_csv, class_to_idx_path, image_root)
    with open(out_dir / "dataset_manifest_audit.json", "w", encoding="utf-8") as f:
        json.dump(manifest_audit, f, indent=2)

    train_df, val_df, _ = load_split_frames(
        split_csv, class_to_idx=class_to_idx, allow_test=False
    )

    per_seed_results: list[dict[str, Any]] = []
    recipe_runs: dict[str, list[dict[str, Any]]] = {r: [] for r in recipes}

    best_overall_macro_f1 = -1.0
    best_overall_run: Optional[dict[str, Any]] = None

    for recipe in recipes:
        for seed in seeds:
            res = train_one_recipe_seed(
                recipe_name=recipe,
                seed=seed,
                train_df=train_df,
                val_df=val_df,
                idx_to_class=idx_to_class,
                image_root=image_root,
                output_dir=out_dir,
                epochs=epochs,
                train_batch_size=batch_size,
                val_batch_size=VAL_BATCH_SIZE,
                num_workers=num_workers,
            )
            recipe_runs[recipe].append(res)

            row = {
                "recipe_name": recipe,
                "seed": seed,
                "best_epoch": res["best_epoch"],
                "macro_f1": res["metrics"]["macro_f1"],
                "accuracy": res["metrics"]["accuracy"],
                "macro_precision": res["metrics"]["macro_precision"],
                "macro_recall": res["metrics"]["macro_recall"],
                "weighted_f1": res["metrics"]["weighted_f1"],
                "val_loss": res["metrics"]["validation_loss"],
                "train_loss": res["metrics"]["train_loss"],
            }
            per_seed_results.append(row)

            if res["metrics"]["macro_f1"] > best_overall_macro_f1:
                best_overall_macro_f1 = res["metrics"]["macro_f1"]
                best_overall_run = res

    # Save per-seed results CSV
    per_seed_df = pd.DataFrame(per_seed_results)
    per_seed_df.to_csv(out_dir / "per_seed_results.csv", index=False)

    # Compute mean and standard deviation per recipe
    mean_std_rows: list[dict[str, Any]] = []
    for recipe, runs in recipe_runs.items():
        f1_scores = [r["metrics"]["macro_f1"] for r in runs]
        seed_f1_dict = {str(r["seed"]): r["metrics"]["macro_f1"] for r in runs}

        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores, ddof=1)) if len(f1_scores) > 1 else 0.0

        mean_std_rows.append({
            "recipe_name": recipe,
            "seed_42_macro_f1": seed_f1_dict.get("42", np.nan),
            "seed_123_macro_f1": seed_f1_dict.get("123", np.nan),
            "seed_2026_macro_f1": seed_f1_dict.get("2026", np.nan),
            "mean_macro_f1": round(mean_f1, 6),
            "std_macro_f1": round(std_f1, 6),
            "num_seeds": len(f1_scores),
        })

    mean_std_df = pd.DataFrame(mean_std_rows).sort_values(by="mean_macro_f1", ascending=False).reset_index(drop=True)
    mean_std_df["rank"] = range(1, len(mean_std_df) + 1)
    mean_std_df.to_csv(out_dir / "mean_std_results.csv", index=False)

    print("\n=======================================================")
    print("Augmentation Recipe Comparison (Ranked by Mean Macro-F1)")
    print("=======================================================")
    print(mean_std_df.to_string(index=False))

    return {
        "per_seed_df": per_seed_df,
        "mean_std_df": mean_std_df,
        "best_overall_run": best_overall_run,
        "manifest_audit": manifest_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ViT-Tiny on PlantVillage.")
    parser.add_argument("--split-csv", type=str, default=str(DEFAULT_SPLIT_CSV), help="Split manifest CSV")
    parser.add_argument("--class-to-idx", type=str, default=str(DEFAULT_CLASS_TO_IDX), help="Class to index mapping")
    parser.add_argument("--image-root", type=str, default=None, help="Root folder of PlantVillage color images")
    parser.add_argument("--output-dir", type=str, default="./vit_tiny_runs", help="Output directory")
    parser.add_argument("--recipes", nargs="+", default=list(AUGMENTATION_RECIPES.keys()), help="Recipes to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="Random seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--dry-run", action="store_true", help="Quick sanity check on a small subset")
    args = parser.parse_args()

    run_full_study(
        recipes=args.recipes,
        seeds=args.seeds,
        split_csv=args.split_csv,
        class_to_idx_path=args.class_to_idx,
        image_root=args.image_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
