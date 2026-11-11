from __future__ import annotations

from terravista_bootstrap import apply_kaggle_triton_guard
apply_kaggle_triton_guard()

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from augmentation_pipeline import (
    AUGMENTATION_RECIPES,
    augment_batch_gpu,
    get_augmentation_recipe,
    zero_probability_config,
)
from terravista_dataset import make_plantvillage_datasets
from terravista_resnet50_settings import (
    BACKBONE_LR,
    CLASSIFIER_LR,
    COMPARISON_EPOCHS,
    NUM_CLASSES,
    SEEDS,
    WEIGHT_DECAY,
    autocast_context,
    build_resnet50,
    environment_report,
    make_criterion,
    make_dataloader_kwargs,
    make_grad_scaler,
    make_optimizer,
    model_settings_dict,
    resolve_device,
    seed_everything,
)
from terravista_training_example import train_one_epoch

PUBLISHED_CE_SEED_SCORES = {
    "no_augmentation": {42: 0.9853, 123: 0.9850, 2026: 0.9837},
    "uniform_p010": {42: 0.9820, 123: 0.9839, 2026: 0.9864},
    "uniform_p020": {42: 0.9820, 123: 0.9842, 2026: 0.9852},
    "joint_refined": {42: 0.9832, 123: 0.9854, 2026: 0.9806},
    "uniform_p025": {42: 0.9814, 123: 0.9804, 2026: 0.9852},
}

PUBLISHED_FROZEN_CE_SEED_SCORES = {
    "uniform_p010": {42: 0.9453, 123: 0.9424, 2026: 0.9464},
}

HEAD_PREFIX = {"resnet50": "fc.", "vit_tiny": "head."}
VIT_TINY_NAME = "vit_tiny_patch16_224"


def build_model(model_name, device):
    if model_name == "resnet50":
        return build_resnet50(device=device, pretrained=True)
    import timm
    model = timm.create_model(VIT_TINY_NAME, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)


def freeze_backbone(model, head_prefix):
    head_name = head_prefix.rstrip(".")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(head_prefix)
    original_train = model.train

    def train(mode=True):
        original_train(mode)
        for name, child in model.named_children():
            if name != head_name:
                child.eval()
        return model

    model.train = train
    return model


def make_model_optimizer(model_name, model):
    if model_name == "resnet50":
        return make_optimizer(model)
    prefix = HEAD_PREFIX[model_name]
    head = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith(prefix) and p.requires_grad]
    groups = [{"params": head, "lr": CLASSIFIER_LR, "weight_decay": WEIGHT_DECAY}]
    if rest:
        groups.insert(0, {"params": rest, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY})
    return torch.optim.AdamW(groups)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits.float(), dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        return (-((1.0 - pt) ** self.gamma) * log_pt).mean()


def parse_args():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--color-dir", required=True)
    parser.add_argument("--model", default="resnet50", choices=list(HEAD_PREFIX))
    parser.add_argument("--recipe", default="uniform_p010", choices=list(AUGMENTATION_RECIPES))
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=COMPARISON_EPOCHS)
    parser.add_argument("--include-ce", action="store_true")
    parser.add_argument("--split-csv", default=str(base / "train_val_split.csv"))
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default="./terravista_cache")
    parser.add_argument("--reuse-cache-dir", nargs="*", default=[])
    parser.add_argument("--allow-different-split", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        suffix = "_frozen" if args.frozen else ""
        if args.model == "resnet50":
            args.output_dir = f"./focal_runs{suffix}"
        else:
            args.output_dir = f"./focal_runs_{args.model}{suffix}"
    return args


def verify_inputs(base: Path, split_csv: str, allow_different_split: bool):
    class_map_path = base / "class_to_idx.json"
    class_to_idx = json.loads(class_map_path.read_text(encoding="utf-8"))
    manifest = json.loads((base / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(class_map_path.read_bytes()).hexdigest() != manifest["dataset"]["class_mapping_sha256"]:
        raise RuntimeError("Bundled class_to_idx.json does not match reproducibility_manifest.json.")
    split_sha = hashlib.sha256(Path(split_csv).read_bytes()).hexdigest()
    if split_sha != manifest["dataset"]["train_val_manifest_sha256"] and not allow_different_split:
        raise RuntimeError("The split CSV does not match the bundled manifest. Use the bundled train_val_split.csv.")
    return class_to_idx, split_sha


@torch.inference_mode()
def predict_validation(model, loader, device):
    model.eval()
    no_augmentation = zero_probability_config()
    true_labels, predictions = [], []
    for images_uint8, labels_cpu in loader:
        images = augment_batch_gpu(images_uint8, no_augmentation, device=device, background_alpha_uint8=None)
        with autocast_context(device):
            logits = model(images)
        true_labels.append(labels_cpu.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(true_labels), np.concatenate(predictions)


def run_one_setting(train_dataset, val_dataset, *, model_name, loss_name, gamma, seed, recipe_name, epochs,
                    class_names, small_classes, output_dir, split_sha, save_checkpoint, frozen=False):
    run_dir = Path(output_dir) / f"{loss_name}_seed{seed}"
    summary_file = run_dir / "run_summary.json"
    if summary_file.is_file():
        previous = json.loads(summary_file.read_text(encoding="utf-8"))
        if (previous.get("loss_name") == loss_name and previous.get("seed") == seed
                and previous.get("recipe_name") == recipe_name and previous.get("epochs") == epochs
                and previous.get("split_sha256") == split_sha
                and previous.get("model", "resnet50") == model_name
                and bool(previous.get("frozen", False)) == bool(frozen)):
            print(f"Skipping completed run: {loss_name}, seed={seed}")
            return previous["result_row"]

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = resolve_device()
    recipe = get_augmentation_recipe(recipe_name)
    train_loader = DataLoader(train_dataset, **make_dataloader_kwargs(seed=seed, training=True))
    val_loader = DataLoader(val_dataset, **make_dataloader_kwargs(seed=seed, training=False))
    model = build_model(model_name, device)
    if frozen:
        freeze_backbone(model, HEAD_PREFIX[model_name])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{model_name} | trainable parameters: {trainable:,} of {total:,}")
    optimizer = make_model_optimizer(model_name, model)
    criterion = FocalLoss(gamma) if gamma > 0 else make_criterion()
    scaler = make_grad_scaler(device)

    labels = list(range(len(class_names)))
    best = {"macro_f1": -1.0}
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, criterion, recipe, device)
        y_true, y_pred = predict_validation(model, val_loader, device)
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))
        history.append({"loss_name": loss_name, "seed": seed, "epoch": epoch,
                        **train_metrics, "val_accuracy": accuracy, "val_macro_f1": macro_f1})
        print(f"{loss_name} | seed={seed} | epoch={epoch}/{epochs} | "
              f"train loss={train_metrics['train_loss']:.4f} | "
              f"val macro-F1={macro_f1:.6f} | val accuracy={accuracy:.6f}")
        if macro_f1 > best["macro_f1"]:
            best = {"macro_f1": macro_f1, "accuracy": accuracy, "epoch": epoch,
                    "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)}
            if save_checkpoint:
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "loss_name": loss_name, "seed": seed}, run_dir / "best_checkpoint.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame({"class": class_names, "f1": best["per_class_f1"]}).to_csv(run_dir / "per_class_f1.csv", index=False)

    row = {"model": model_name, "loss_name": loss_name, "gamma": gamma, "seed": seed, "recipe_name": recipe_name,
           "epochs": epochs, "best_epoch": best["epoch"],
           "val_macro_f1": best["macro_f1"], "val_accuracy": best["accuracy"]}
    for class_name in small_classes:
        row[f"f1_{class_name}"] = float(best["per_class_f1"][class_names.index(class_name)])
    summary = {"model": model_name, "loss_name": loss_name, "gamma": gamma, "seed": seed,
               "recipe_name": recipe_name, "epochs": epochs, "split_sha256": split_sha,
               "frozen": bool(frozen), "trainable_parameters": trainable, "total_parameters": total,
               "result_row": row,
               "model_settings": {**model_settings_dict(), "all_layers_trainable": not frozen},
               "environment": environment_report()}
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return row


def build_comparison(per_seed, small_classes, recipe_name, epochs, include_ce, frozen=False, model_name="resnet50"):
    rows = []
    for loss_name, group in per_seed.groupby("loss_name", sort=False):
        scores = group["val_macro_f1"].to_numpy(dtype=float)
        row = {"loss": loss_name, "n_seeds": len(scores),
               "val_macro_f1_mean": float(scores.mean()),
               "val_macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0}
        for class_name in small_classes:
            row[f"f1_{class_name}"] = float(group[f"f1_{class_name}"].mean())
        row["f1_five_smallest_mean"] = float(np.mean([row[f"f1_{c}"] for c in small_classes]))
        rows.append(row)

    if model_name == "resnet50":
        published = PUBLISHED_FROZEN_CE_SEED_SCORES if frozen else PUBLISHED_CE_SEED_SCORES
        reference = published.get(recipe_name)
        if not include_ce and reference and epochs == COMPARISON_EPOCHS:
            scores = np.array(list(reference.values()), dtype=float)
            label = "cross_entropy (team frozen study, 3 epochs)" if frozen else "cross_entropy (team study, 3 epochs)"
            row = {"loss": label, "n_seeds": len(scores),
                   "val_macro_f1_mean": float(scores.mean()),
                   "val_macro_f1_std": float(scores.std(ddof=1))}
            for class_name in small_classes:
                row[f"f1_{class_name}"] = float("nan")
            row["f1_five_smallest_mean"] = float("nan")
            rows.insert(0, row)
    return pd.DataFrame(rows)


def draft_conclusion(comparison, small_classes):
    is_ce = comparison["loss"].str.startswith("cross_entropy")
    if not is_ce.any():
        return "No cross-entropy reference available for comparison."
    ce = comparison[is_ce].iloc[0]
    lines = []
    for _, row in comparison[~is_ce].iterrows():
        delta = row["val_macro_f1_mean"] - ce["val_macro_f1_mean"]
        spread = max(row["val_macro_f1_std"], ce["val_macro_f1_std"])
        if abs(delta) <= spread:
            verdict = "within the seed-to-seed spread, so there is no clear difference"
        elif delta > 0:
            verdict = "above the seed-to-seed spread"
        else:
            verdict = "below the seed-to-seed spread"
        line = (f"{row['loss']}: mean validation macro-F1 {row['val_macro_f1_mean']:.4f} "
                f"(std {row['val_macro_f1_std']:.4f}) vs {ce['val_macro_f1_mean']:.4f} "
                f"(std {ce['val_macro_f1_std']:.4f}) for cross-entropy; difference {delta:+.4f}, {verdict}.")
        if np.isfinite(ce["f1_five_smallest_mean"]) and np.isfinite(row["f1_five_smallest_mean"]):
            small_delta = row["f1_five_smallest_mean"] - ce["f1_five_smallest_mean"]
            line += (f" Mean F1 over the five smallest classes: {row['f1_five_smallest_mean']:.4f} "
                     f"vs {ce['f1_five_smallest_mean']:.4f} ({small_delta:+.4f}).")
        lines.append(line)
    lines.append("Draft generated from the numbers above. Check it and edit before submitting.")
    return "\n".join(lines)


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent
    device = resolve_device()
    if device.type != "cuda":
        raise RuntimeError("This run needs a CUDA GPU. Select a GPU in Kaggle.")

    class_to_idx, split_sha = verify_inputs(base, args.split_csv, args.allow_different_split)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    need_masks = AUGMENTATION_RECIPES[args.recipe]["p_background"] > 0
    train_ds, val_ds, train_df, val_df, mask_path, image_cache_path, cache_fingerprint = make_plantvillage_datasets(
        args.split_csv, args.color_dir, class_to_idx,
        need_background_masks=need_masks, cache_dir=args.cache_dir,
        reuse_cache_dirs=args.reuse_cache_dir,
    )
    print(f"Train={len(train_ds):,}; validation={len(val_ds):,}; classes={len(class_names)}; recipe={args.recipe}; "
          f"model={args.model}; mode={'frozen backbone, head only' if args.frozen else 'all layers trainable'}")
    print("Test images: not resolved, loaded, or evaluated.")

    counts = train_df["class"].astype(str).value_counts().sort_values(kind="stable")
    small_classes = list(counts.index[:5])
    print("Five smallest classes (train images):", {name: int(counts[name]) for name in small_classes})

    settings = []
    if args.include_ce:
        settings.append(("cross_entropy", 0.0))
    settings += [(f"focal_gamma{gamma:g}", gamma) for gamma in args.gammas]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        "model": args.model, "recipe": args.recipe, "gammas": args.gammas, "seeds": args.seeds,
        "epochs": args.epochs, "include_ce": args.include_ce, "frozen": args.frozen,
        "split_sha256": split_sha, "five_smallest_classes": small_classes,
        "model_settings": model_settings_dict(), "environment": environment_report(),
        "test_data_used": False,
    }, indent=2), encoding="utf-8")

    rows = []
    for loss_name, gamma in settings:
        for seed in args.seeds:
            row = run_one_setting(
                train_ds, val_ds, model_name=args.model, loss_name=loss_name, gamma=gamma, seed=seed,
                recipe_name=args.recipe, epochs=args.epochs, class_names=class_names,
                small_classes=small_classes, output_dir=output_dir, split_sha=split_sha,
                save_checkpoint=args.save_checkpoints, frozen=args.frozen,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / "per_seed_results.csv", index=False)

    per_seed = pd.DataFrame(rows)
    comparison = build_comparison(per_seed, small_classes, args.recipe, args.epochs, args.include_ce,
                                  args.frozen, args.model)
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)
    conclusion = draft_conclusion(comparison, small_classes)
    (output_dir / "conclusion_draft.txt").write_text(conclusion + "\n", encoding="utf-8")

    print("\n=== COMPARISON (validation only) ===")
    print(comparison.round(4).to_string(index=False))
    print("\n" + conclusion)
    print(f"\nSaved to {output_dir}/: per_seed_results.csv, comparison_table.csv, conclusion_draft.txt")


if __name__ == "__main__":
    main()
