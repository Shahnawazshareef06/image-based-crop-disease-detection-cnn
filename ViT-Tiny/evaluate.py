"""Comprehensive evaluation and error analysis for ViT-Tiny on PlantVillage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from config import (
    DEFAULT_CLASS_TO_IDX,
    DEFAULT_SPLIT_CSV,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)
from dataset import PlantVillageDataset, load_split_frames


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    loss: float = 0.0,
    train_loss: float = 0.0,
) -> dict[str, float]:
    """Compute all required project validation metrics."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_p = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    return {
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "validation_loss": round(loss, 6),
        "train_loss": round(train_loss, 6),
    }


def generate_classification_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    idx_to_class: Mapping[int, str],
) -> pd.DataFrame:
    """Generate per-class precision, recall, f1-score, and support table."""
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[i] for i in labels]

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    df = pd.DataFrame({
        "class_id": labels,
        "class_name": target_names,
        "precision": np.round(p, 6),
        "recall": np.round(r, 6),
        "f1_score": np.round(f1, 6),
        "support": s.astype(int),
    })
    return df


def find_weakest_classes(
    per_class_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Extract top-k weakest performing classes sorted by lowest F1-score."""
    return per_class_df.sort_values(by="f1_score", ascending=True).head(top_k).reset_index(drop=True)


def find_most_confused_pairs(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    idx_to_class: Mapping[int, str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Identify the most frequent off-diagonal class confusion pairs."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    num_classes = len(idx_to_class)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))
    np.fill_diagonal(cm, 0)  # Ignore correct classifications

    pairs: list[dict[str, Any]] = []
    indices = np.argwhere(cm > 0)
    for true_idx, pred_idx in indices:
        count = int(cm[true_idx, pred_idx])
        pairs.append({
            "actual_class": idx_to_class[true_idx],
            "predicted_class": idx_to_class[pred_idx],
            "actual_id": int(true_idx),
            "predicted_id": int(pred_idx),
            "confusion_count": count,
        })

    df = pd.DataFrame(pairs)
    if not df.empty:
        df = df.sort_values(by="confusion_count", ascending=False).head(top_k).reset_index(drop=True)
    return df


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    idx_to_class: Mapping[int, str],
    save_path: str | Path,
    normalize: bool = False,
    title: str = "ViT-Tiny Validation Confusion Matrix",
) -> None:
    """Render and save a high-resolution confusion matrix heatmap."""
    num_classes = len(idx_to_class)
    labels = list(range(num_classes))
    class_names = [idx_to_class[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm_float = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2f"
        data_to_plot = cm_float
    else:
        fmt = "d"
        data_to_plot = cm

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        data_to_plot,
        annot=False,  # 38 classes is dense, heatmap shading provides clear visual pattern
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_predictions_dataframe(
    paths: Sequence[str],
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    idx_to_class: Mapping[int, str],
) -> pd.DataFrame:
    """Generate structured prediction CSV for downstream multi-model comparison."""
    df = pd.DataFrame({
        "image_path": paths,
        "actual_class": [idx_to_class[int(y)] for y in y_true],
        "predicted_class": [idx_to_class[int(y)] for y in y_pred],
        "actual_id": [int(y) for y in y_true],
        "predicted_id": [int(y) for y in y_pred],
        "is_correct": [int(t == p) for t, p in zip(y_true, y_pred)],
    })
    return df


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Run full evaluation on a PyTorch dataloader and collect targets and predictions."""
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    for batch in dataloader:
        images_uint8 = batch[0].to(device)
        targets = batch[1].to(device)

        # Convert uint8 NHWC to float NCHW [0, 1] then ImageNet normalize
        images = images_uint8.permute(0, 3, 1, 2).to(torch.float32) / 255.0
        images = (images - mean) / std

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, all_targets, all_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ViT-Tiny model on PlantVillage.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--split-csv", type=str, default=str(DEFAULT_SPLIT_CSV), help="Dataset split CSV")
    parser.add_argument("--class-to-idx", type=str, default=str(DEFAULT_CLASS_TO_IDX), help="Class to index mapping")
    parser.add_argument("--image-root", type=str, default=None, help="Root folder of PlantVillage images")
    parser.add_argument("--output-dir", type=str, default="./evaluation_output", help="Directory for evaluation artifacts")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicit flag to evaluate the test split. Strict restriction: do not evaluate test during tuning.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.class_to_idx, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_df, val_df, test_df = load_split_frames(
        args.split_csv, class_to_idx=class_to_idx, allow_test=args.evaluate_test
    )

    eval_df = test_df if (args.evaluate_test and test_df is not None) else val_df
    split_name = "test" if args.evaluate_test else "val"
    print(f"Evaluating split: {split_name} ({len(eval_df)} images)")

    from model import build_vit_tiny

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit_tiny(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    dataset = PlantVillageDataset(eval_df, image_root=args.image_root)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    loss, y_true, y_pred = evaluate_model(model, dataloader, criterion, device)
    metrics = compute_metrics(y_true, y_pred, loss=loss)
    print("Metrics:", json.dumps(metrics, indent=2))

    # Error analysis
    report_df = generate_classification_report(y_true, y_pred, idx_to_class)
    report_df.to_csv(out_dir / "classification_report.csv", index=False)

    weakest_df = find_weakest_classes(report_df, top_k=5)
    weakest_df.to_csv(out_dir / "weakest_classes.csv", index=False)

    confused_df = find_most_confused_pairs(y_true, y_pred, idx_to_class, top_k=10)
    confused_df.to_csv(out_dir / "confused_class_pairs.csv", index=False)

    preds_df = generate_predictions_dataframe(eval_df["relative_path"].tolist(), y_true, y_pred, idx_to_class)
    preds_df.to_csv(out_dir / "validation_predictions.csv", index=False)

    misclassified_df = preds_df[preds_df["is_correct"] == 0]
    misclassified_df.to_csv(out_dir / "misclassified_images.csv", index=False)

    plot_confusion_matrix(y_true, y_pred, idx_to_class, out_dir / "confusion_matrix.png")
    print(f"Evaluation complete. Artifacts saved in {out_dir}")


if __name__ == "__main__":
    main()
