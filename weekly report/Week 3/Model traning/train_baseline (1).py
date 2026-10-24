"""
Part 4 - Baseline: frozen ImageNet ResNet-50 + new classifier head (PyTorch).

Input : split.csv with columns  path, class, split   (split in {train, val, test})
Output: accuracy, per-class F1, confusion matrix on val and test (+ training curve).

- Backbone fully frozen and kept in eval mode (BatchNorm stats never change).
- Default ("cached") mode: no augmentation, 2048-d features extracted once, head trained
  on the cached features (fast; identical to training the head on a frozen ResNet).
- --augment: train images are augmented on the fly with augmentation_pipeline.py (TRAIN only),
  backbone still frozen. Val/test are never augmented.
- Model selection: epoch with best VAL macro-F1. TEST is evaluated once at the end.

Usage:
  python train_baseline.py --csv split.csv --root /path/to/color --out results
  python train_baseline.py --csv split.csv --root /path/to/dataset --augment --out results_aug
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

from augmentation_pipeline import build_eval_transform, build_train_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="split.csv (path, class, split)")
    p.add_argument("--root", default="", help="Prefix added to relative paths in the csv")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=None,
                   help="default 30 (cached) or 10 (--augment)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true",
                   help="augment train images on the fly (augmentation_pipeline.py)")
    p.add_argument("--bg-prob", type=float, default=0.0,
                   help="Prob. of synthetic background replacement (GrabCut, slow). "
                        "Keep 0 until the masks are visually checked.")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="Stratified fraction of train to use (data-efficiency runs)")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                   help="'none' is only for smoke-testing")
    return p.parse_args()


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---- data -------------------------------------------------------------------
class CsvDataset(Dataset):
    def __init__(self, df, class_to_idx, transform, root=""):
        self.paths = [str(Path(root) / p) if root else p for p in df["path"]]
        self.labels = [class_to_idx[c] for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    need = {"path", "class", "split"}
    if not need.issubset(df.columns):
        raise ValueError(f"split.csv needs columns {need}, found {list(df.columns)}")
    df["split"] = df["split"].str.strip().str.lower().replace(
        {"valid": "val", "validation": "val"})
    bad = set(df["split"]) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"Unknown split values: {bad}")
    return df


def check_paths(df, root):
    missing = [p for p in df["path"] if not (Path(root) / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(df)} images not found under --root '{root}'. "
            f"First missing: {missing[0]}")


# ---- model ------------------------------------------------------------------
def build_backbone(weights, device):
    w = models.ResNet50_Weights.IMAGENET1K_V2 if weights == "imagenet" else None
    m = models.resnet50(weights=w)
    m.fc = nn.Identity()                  # 2048-d pooled features
    for prm in m.parameters():
        prm.requires_grad = False         # frozen
    return m.eval().to(device)            # eval(): BN uses ImageNet running stats


def make_head(n_classes):
    return nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, n_classes))


def feats_of(backbone, x, device):
    with torch.no_grad(), torch.autocast(device_type=device.type,
                                         enabled=device.type == "cuda"):
        return backbone(x.to(device, non_blocking=True)).float()


def extract(backbone, ds, device, args, desc):
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    fs, ys = [], []
    for x, y in tqdm(loader, desc=desc):
        fs.append(feats_of(backbone, x, device).cpu())
        ys.append(y)
    return torch.cat(fs), torch.cat(ys)


@torch.no_grad()
def predict(head, feats, device, batch=2048):
    head.eval()
    return torch.cat([head(feats[i:i + batch].to(device)).argmax(1).cpu()
                      for i in range(0, len(feats), batch)]).numpy()


# ---- evaluation -------------------------------------------------------------
def evaluate(y_true, y_pred, names):
    labels = list(range(len(names)))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                    zero_division=0)
    per_class = pd.DataFrame({"class": names, "precision": p, "recall": r,
                              "f1": f1, "support": sup})
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    summ = {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels,
                                       zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                          labels=labels, zero_division=0)),
            "n_samples": int(len(y_true))}
    rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                digits=4, zero_division=0)
    return summ, per_class, cm, rep


def plot_cm(cm, names, title, path):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35),) * 2)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    epochs = args.epochs or (10 if args.augment else 30)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| mode:", "augment" if args.augment else "cached")

    df = load_split(args.csv)
    check_paths(df, args.root)
    names = sorted(df["class"].unique())
    c2i = {c: i for i, c in enumerate(names)}
    parts = {k: df[df.split == k].reset_index(drop=True) for k in ("train", "val", "test")}
    if args.train_frac < 1.0:
        keep, _ = train_test_split(np.arange(len(parts["train"])), train_size=args.train_frac,
                                   stratify=parts["train"]["class"], random_state=args.seed)
        parts["train"] = parts["train"].iloc[np.sort(keep)].reset_index(drop=True)
    print(f"{len(names)} classes | " + " | ".join(f"{k} {len(v)}" for k, v in parts.items()))
    missing = set(names) - set(parts["train"]["class"])
    if missing:
        raise ValueError(f"Classes with no train images: {sorted(missing)}")

    (out / "classes.txt").write_text("\n".join(names))
    parts["train"]["class"].value_counts().reindex(names).rename("train_images").to_csv(
        out / "train_class_counts.csv")          # handy for the imbalance work (part 5)

    backbone = build_backbone(args.weights, device)
    t0 = time.time()
    Xva, yva = extract(backbone, CsvDataset(parts["val"], c2i, build_eval_transform(normalize=True), args.root),
                       device, args, "features: val")
    Xte, yte = extract(backbone, CsvDataset(parts["test"], c2i, build_eval_transform(normalize=True), args.root),
                       device, args, "features: test")
    if args.augment:
        train_ds_aug = CsvDataset(
            parts["train"], c2i,
            build_train_transform(normalize=True, background_probability=args.bg_prob),
            args.root)
        train_loader = DataLoader(train_ds_aug, batch_size=args.batch, shuffle=True,
                                  num_workers=args.workers, pin_memory=device.type == "cuda",
                                  drop_last=False)
    else:
        Xtr, ytr = extract(backbone, CsvDataset(parts["train"], c2i, build_eval_transform(normalize=True), args.root),
                           device, args, "features: train")
        Xtr, ytr = Xtr.to(device), ytr.to(device)
    print(f"Feature extraction done in {time.time() - t0:.0f}s")

    head = make_head(len(names)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history, best_f1, best_ep = [], -1.0, -1
    for ep in range(1, epochs + 1):
        head.train()
        total, n = 0.0, 0
        if args.augment:
            for x, y in tqdm(train_loader, desc=f"epoch {ep}", leave=False):
                f, y = feats_of(backbone, x, device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(head(f), y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
                n += len(y)
        else:
            perm = torch.randperm(len(Xtr), device=device)
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                opt.zero_grad()
                loss = loss_fn(head(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
                n += len(b)
        sched.step()
        vp = predict(head, Xva, device)
        v_acc = accuracy_score(yva.numpy(), vp)
        v_f1 = f1_score(yva.numpy(), vp, average="macro", zero_division=0)
        history.append({"epoch": ep, "train_loss": total / n, "val_acc": v_acc,
                        "val_macro_f1": v_f1})
        flag = ""
        if v_f1 > best_f1:
            best_f1, best_ep = v_f1, ep
            torch.save({"head": head.state_dict(), "classes": names}, out / "head_best.pt")
            flag = "  *best*"
        print(f"epoch {ep:02d} loss {total / n:.4f} val_acc {v_acc:.4f} "
              f"val_macro_f1 {v_f1:.4f}{flag}")

    h = pd.DataFrame(history)
    h.to_csv(out / "training_curve.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss)
    ax[0].set_title("Train loss")
    ax[1].plot(h.epoch, h.val_acc, label="val acc")
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=150)
    plt.close(fig)

    head.load_state_dict(torch.load(out / "head_best.pt", map_location=device)["head"])
    results = {"mode": "augment" if args.augment else "cached", "best_epoch": best_ep,
               "epochs": epochs, "n_classes": len(names), "seed": args.seed,
               "train_frac": args.train_frac,
               "split_sizes": {k: int(len(v)) for k, v in parts.items()}}
    for name, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        yp = predict(head, X, device)
        summ, per_class, cm, rep = evaluate(y.numpy(), yp, names)
        results[name] = summ
        per_class.to_csv(out / f"{name}_per_class.csv", index=False)
        pd.DataFrame(cm, index=names, columns=names).to_csv(
            out / f"{name}_confusion_matrix.csv")
        (out / f"{name}_report.txt").write_text(rep)
        plot_cm(cm, names, f"Confusion matrix ({name}, row-normalised)",
                out / f"{name}_confusion_matrix.png")
        print(f"\n=== {name.upper()} === acc {summ['accuracy']:.4f} | "
              f"macro-F1 {summ['macro_f1']:.4f} | weighted-F1 {summ['weighted_f1']:.4f}")
        print("Lowest-F1 classes:\n" + per_class.sort_values("f1").head(5)[
            ["class", "f1", "support"]].to_string(index=False))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nBest epoch (val macro-F1): {best_ep}. Saved to {out}/")


if __name__ == "__main__":
    main()
