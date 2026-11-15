
from __future__ import annotations


from terravista_bootstrap import apply_kaggle_triton_guard
apply_kaggle_triton_guard()

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import build_mobilenetv3, make_optimizer
from augmentation_pipeline import (
    AUGMENTATION_RECIPES,
    augment_batch_gpu,
    get_augmentation_recipe,
    zero_probability_config,
)
from dataset import make_plantvillage_datasets, normalize_relative_path
from config import (
    BACKBONE_LR,
    CLASSIFIER_LR,
    COMPARISON_EPOCHS,
    IMAGE_SIZE,
    NUM_CLASSES,
    SEEDS,
    TRAIN_BATCH_SIZE,
    VALIDATION_BATCH_SIZE,
    WEIGHT_DECAY,
    autocast_context,
    environment_report,
    make_criterion,
    make_dataloader_kwargs,
    make_grad_scaler,
    resolve_device,
    seed_everything,
)
from engine import evaluate_validation, train_one_epoch




@torch.inference_mode()
def predict(model, loader, device):
    """Deterministic predictions (all augmentation probabilities zero)."""
    model.eval()
    no_aug = zero_probability_config()
    ys, ps = [], []
    for images_uint8, labels in loader:
        images = augment_batch_gpu(images_uint8, no_aug, device=device, background_alpha_uint8=None)
        with autocast_context(device):
            logits = model(images)
        ys.append(labels.numpy())
        ps.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.inference_mode()
def gpu_ms_per_image(model, sample_uint8, device, iters=30, warmup=5):
    """Forward-pass time only: GPU, batch 64, FP16 autocast, no data loading."""
    model.eval()
    x = augment_batch_gpu(sample_uint8, zero_probability_config(), device=device)
    for _ in range(warmup):
        with autocast_context(device):
            model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with autocast_context(device):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (iters * x.shape[0]) * 1000.0


@torch.inference_mode()
def cpu_ms_per_image(model, iters=30, warmup=5):
    """Forward-pass time only: CPU, 1 thread, batch 1, FP32."""
    cpu_model = copy.deepcopy(model).to("cpu").float().eval()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    for _ in range(warmup):
        cpu_model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        cpu_model(x)
    return (time.perf_counter() - t0) / iters * 1000.0


class TestImages(Dataset):
    """Reads test images directly with the same PIL bilinear 224px resize as the cache."""
    def __init__(self, rel_paths, labels, color_dir):
        self.rel_paths, self.labels, self.root = list(rel_paths), list(labels), Path(color_dir)

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.rel_paths[i]) as im:
            arr = np.asarray(
                im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        return torch.from_numpy(arr.copy()), int(self.labels[i])


def run_final_test(model, args, class_to_idx, train_df, val_df, run_dir, seed, device):
    full = pd.read_csv(args.full_split_csv)
    full["split"] = full["split"].astype(str).str.strip().str.lower()
    test_df = full[full["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        raise ValueError("No test rows in --full-split-csv.")
    test_df["relative_path"] = test_df["path"].map(normalize_relative_path)
    used = set(train_df["relative_path"]) | set(val_df["relative_path"])
    if used & set(test_df["relative_path"]):
        raise RuntimeError("Leakage: test paths overlap train/val paths.")
    labels = test_df["class"].astype(str).map(class_to_idx).to_numpy()
    loader = DataLoader(
        TestImages(test_df["relative_path"], labels, args.color_dir),
        batch_size=VALIDATION_BATCH_SIZE, shuffle=False, num_workers=2,
    )
    y_true, y_pred = predict(model, loader, device)
    result = {
        "seed": seed,
        "test_rows": int(len(y_true)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                        average="macro", zero_division=0)),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL TEST (use once, final model only):", result)
    return result


def parse_args():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="MobileNetV3 under the TerraVista protocol")
    p.add_argument("--color-dir", required=True, help="PlantVillage 'color' directory")
    p.add_argument("--split-csv", default=str(base / "train_val_split.csv"))
    p.add_argument("--output-dir", default="./results/mobilenetv3")
    p.add_argument("--cache-dir", default="./terravista_cache")
    p.add_argument("--reuse-cache-dir", nargs="*", default=[],
                   help="read-only prior cache folder(s), e.g. a Kaggle attached Output/cache directory")
    p.add_argument("--variant", choices=["large", "small"], default="large")
    p.add_argument("--recipe", default="no_augmentation", choices=list(AUGMENTATION_RECIPES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=COMPARISON_EPOCHS)
    p.add_argument("--force", action="store_true", help="rerun seeds that already have metrics.json")
    p.add_argument("--allow-different-split", action="store_true")
    p.add_argument("--final-test", action="store_true",
                   help="ALSO score the test split. Single seed, final model, once. Needs --full-split-csv")
    p.add_argument("--full-split-csv", default=None, help="split.csv that includes the test rows")
    args = p.parse_args()
    if args.final_test and (len(args.seeds) != 1 or not args.full_split_csv):
        p.error("--final-test needs exactly one seed in --seeds and --full-split-csv.")
    return args


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent
    device = resolve_device()
    if device.type != "cuda":
        raise RuntimeError("Select a CUDA GPU (the protocol uses FP16 autocast on GPU).")

    class_to_idx = json.loads((base / "class_to_idx.json").read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    manifest = json.loads((base / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    split_sha = hashlib.sha256(Path(args.split_csv).read_bytes()).hexdigest()
    if split_sha != manifest["dataset"]["train_val_manifest_sha256"] and not args.allow_different_split:
        raise RuntimeError("Split CSV does not match the bundled manifest. Use the bundled train_val_split.csv.")

    recipe = get_augmentation_recipe(args.recipe)
    need_masks = recipe["p_background"] > 0
    train_ds, val_ds, train_df, val_df, _mask, image_cache_path, fingerprint = make_plantvillage_datasets(
        args.split_csv, args.color_dir, class_to_idx,
        need_background_masks=need_masks, cache_dir=args.cache_dir,
        reuse_cache_dirs=args.reuse_cache_dir,
    )
    print(f"Train={len(train_ds):,}; validation={len(val_ds):,}; classes={len(class_to_idx)}")
    if not args.final_test:
        print("Test images: not resolved, loaded, or evaluated.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_uint8 = torch.stack([val_ds[i][0] for i in range(VALIDATION_BATCH_SIZE)])

    for seed in args.seeds:
        run_dir = out_dir / f"mobilenetv3_{args.variant}_{args.recipe}_seed{seed}"
        metrics_file = run_dir / "metrics.json"
        if metrics_file.is_file() and not args.force and not args.final_test:
            print(f"Skipping finished run: {run_dir.name}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(seed)
        train_loader = DataLoader(train_ds, **make_dataloader_kwargs(seed=seed, training=True))
        val_loader = DataLoader(val_ds, **make_dataloader_kwargs(seed=seed, training=False))
        model = build_mobilenetv3(args.variant, device)
        optimizer = make_optimizer(model)
        criterion = make_criterion()
        scaler = make_grad_scaler(device)

        weights_path = run_dir / "best_weights.pt"
        best_f1, best_epoch, best_metrics, history = -1.0, 0, None, []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, optimizer, scaler, criterion, recipe, device)
            va = evaluate_validation(model, val_loader, criterion, device)
            row = {"epoch": epoch, **tr, **va}
            history.append(row)
            print(f"mobilenetv3_{args.variant} | {args.recipe} | seed={seed} | epoch={epoch}/{args.epochs} | "
                  f"val macro-F1={va['val_macro_f1']:.6f} | val acc={va['val_accuracy']:.6f} | {time.time()-t0:.0f}s")
            if va["val_macro_f1"] > best_f1:
                best_f1, best_epoch, best_metrics = va["val_macro_f1"], epoch, dict(row)
                torch.save(model.state_dict(), weights_path)

 
        model.load_state_dict(torch.load(weights_path, map_location=device))
        y_true, y_pred = predict(model, val_loader, device)
        pd.DataFrame({
            "path": val_df["path"].to_numpy(),
            "true": [idx_to_class[int(i)] for i in y_true],
            "pred": [idx_to_class[int(i)] for i in y_pred],
        }).to_csv(run_dir / "val_predictions.csv", index=False)
        recomputed_f1 = float(f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                       average="macro", zero_division=0))

        metrics = {
            "model": f"mobilenet_v3_{args.variant}",
            "recipe": args.recipe,
            "augmentation_probabilities": recipe,
            "seed": seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": float(best_f1),
            "val_macro_f1_recomputed_from_predictions": recomputed_f1,
            "best_validation_metrics": best_metrics,
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "weights_file_size_mb": round(weights_path.stat().st_size / 1e6, 2),
            "gpu_ms_per_image_bs64_fp16": round(gpu_ms_per_image(model, sample_uint8, device), 4),
            "cpu_ms_per_image_bs1_fp32_1thread": round(cpu_ms_per_image(model), 3),
            "settings": {
                "pretrained_weights": "DEFAULT (ImageNet)",
                "all_layers_trainable": True,
                "optimizer": "AdamW",
                "backbone_lr": BACKBONE_LR,
                "head_lr": CLASSIFIER_LR,
                "head_definition": "model.classifier[3] (new 38-class layer)",
                "weight_decay": WEIGHT_DECAY,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "validation_batch_size": VALIDATION_BATCH_SIZE,
            },
            "split_sha256": split_sha,
            "image_cache_fingerprint": fingerprint,
            "environment": environment_report(),
            "history": history,
        }
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Seed {seed}: best epoch {best_epoch}, val macro-F1 {best_f1:.6f}")

        if args.final_test:
            run_final_test(model, args, class_to_idx, train_df, val_df, run_dir, seed, device)


if __name__ == "__main__":
    main()