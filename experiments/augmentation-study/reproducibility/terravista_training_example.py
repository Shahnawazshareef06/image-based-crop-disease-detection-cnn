"""Reusable TerraVista training/evaluation functions.

This is a project adapter, not a complete dataset loader. The team member should
pass in the same train/validation Dataset objects and class_to_idx mapping built
from the supplied train_val_split.csv. See README.md for the exact Dataset batch
contract and recipe loop.

Expected training batch:
    (images_uint8, labels) or
    (images_uint8, labels, background_alpha_uint8)

images_uint8 must be RGB uint8 NHWC, resized to 224x224 before this function.
If p_background > 0, masks must be returned as the third item and aligned with
that same training image. Validation batches must be (images_uint8, labels).
"""
from __future__ import annotations

# Apply the Kaggle Triton compatibility guard before importing any PyTorch file.
import sys
from terravista_bootstrap import apply_kaggle_triton_guard
if "torch" not in sys.modules:
    apply_kaggle_triton_guard()

import json
import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from augmentation_pipeline import augment_batch_gpu, get_augmentation_recipe, zero_probability_config
from terravista_resnet50_settings import (
    SEEDS,
    COMPARISON_EPOCHS,
    seed_everything,
    make_dataloader_kwargs,
    build_resnet50,
    make_optimizer,
    make_criterion,
    make_grad_scaler,
    autocast_context,
    resolve_device,
    model_settings_dict,
    environment_report,
)


def _unpack_training_batch(batch):
    if len(batch) == 3:
        images_uint8, labels, alpha_masks = batch
    elif len(batch) == 2:
        images_uint8, labels = batch
        alpha_masks = None
    else:
        raise ValueError("Training batches must be (images, labels[, foreground_alpha_masks]).")
    return images_uint8, labels, alpha_masks


def train_one_epoch(model, loader, optimizer, scaler, criterion, recipe, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        images_uint8, labels_cpu, alpha_masks = _unpack_training_batch(batch)
        if recipe["p_background"] > 0 and alpha_masks is None:
            raise ValueError(
                "The selected recipe uses p_background > 0. The training Dataset must "
                "return a matching uint8 foreground-alpha mask as the third batch item."
            )

        images = augment_batch_gpu(
            images_uint8,
            recipe,
            device=device,
            background_alpha_uint8=alpha_masks,
        )
        labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(labels.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return {
        "train_loss": total_loss / max(total_examples, 1),
        "train_accuracy": total_correct / max(total_examples, 1),
    }


@torch.inference_mode()
def evaluate_validation(model, loader, criterion, device):
    """Evaluate deterministic validation batches with all augmentation p=0."""
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    total_examples = 0
    no_aug = zero_probability_config()

    for batch in loader:
        if len(batch) != 2:
            raise ValueError("Validation Dataset must return only (images_uint8, labels).")
        images_uint8, labels_cpu = batch
        images = augment_batch_gpu(images_uint8, no_aug, device=device, background_alpha_uint8=None)
        labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)

        with autocast_context(device):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_true.append(labels.cpu().numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    labels_range = list(range(len(set(y_true.tolist()))))
    return {
        "val_loss": total_loss / max(total_examples, 1),
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_precision": float(precision_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)),
        "val_macro_recall": float(recall_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)),
        "val_macro_f1": float(f1_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)),
        "val_weighted_f1": float(f1_score(y_true, y_pred, labels=labels_range, average="weighted", zero_division=0)),
    }


def run_one_recipe_seed(
    train_dataset,
    validation_dataset,
    *,
    recipe_name: str,
    seed: int,
    class_to_idx: Mapping[str, int],
    output_dir: str | Path,
    run_metadata: Mapping | None = None,
    device=None,
):
    """Run one complete three-epoch (recipe, seed) experiment and save artifacts.

    Re-run this function separately for each recipe and seed. It resets every
    RNG before model construction and uses a dedicated seeded DataLoader generator.
    """
    if len(class_to_idx) != 38:
        raise ValueError(f"Expected a 38-class mapping; got {len(class_to_idx)} classes.")
    if set(class_to_idx.values()) != set(range(38)):
        raise ValueError("class_to_idx values must be contiguous integers 0..37.")

    seed_everything(seed)
    device = resolve_device(device)
    recipe = get_augmentation_recipe(recipe_name)

    train_loader = DataLoader(
        train_dataset,
        **make_dataloader_kwargs(seed=seed, training=True),
    )
    validation_loader = DataLoader(
        validation_dataset,
        **make_dataloader_kwargs(seed=seed, training=False),
    )

    model = build_resnet50(device=device, pretrained=True)
    optimizer = make_optimizer(model)
    criterion = make_criterion()
    scaler = make_grad_scaler(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"{recipe_name}_seed{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_metrics = None
    history = []

    for epoch in range(1, COMPARISON_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, recipe, device
        )
        val_metrics = evaluate_validation(model, validation_loader, criterion, device)
        row = {"recipe_name": recipe_name, "seed": int(seed), "epoch": epoch,
               **train_metrics, **val_metrics}
        history.append(row)
        print(
            f"{recipe_name} | seed={seed} | epoch={epoch}/{COMPARISON_EPOCHS} | "
            f"val macro-F1={val_metrics['val_macro_f1']:.6f} | "
            f"val accuracy={val_metrics['val_accuracy']:.6f}"
        )

        if val_metrics["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["val_macro_f1"]
            best_metrics = dict(row)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "grad_scaler_state_dict": scaler.state_dict(),
                "best_epoch": epoch,
                "best_validation_metrics": best_metrics,
                "recipe_name": recipe_name,
                "augmentation_probabilities": recipe,
                "seed": int(seed),
                "class_to_idx": dict(class_to_idx),
                "model_settings": model_settings_dict(),
                "environment": environment_report(),
                "run_metadata": dict(run_metadata or {}),
            }
            torch.save(checkpoint, run_dir / "best_checkpoint.pt")

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "recipe_name": recipe_name,
        "seed": int(seed),
        "best_epoch": int(best_metrics["epoch"]),
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_metrics": best_metrics,
        "augmentation_probabilities": recipe,
        "model_settings": model_settings_dict(),
        "environment": environment_report(),
        "class_to_idx": dict(class_to_idx),
        "run_metadata": dict(run_metadata or {}),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print("TerraVista training integration module loaded.")
    print("This file expects project-specific train/validation Dataset objects.")
    print("See README.md for the exact Dataset contract and 5-recipe/3-seed loop.")
    print("Supported recipes:", [
        "no_augmentation", "uniform_p010", "uniform_p020", "joint_refined", "uniform_p025"
    ])
