"""Evaluate saved MobileNetV3-Large checkpoints on the PlantVillage TEST split.

Run from inside MobileNetV3_Handover (train.py, model.py, dataset.py, config.py next to it).
The test set must be used ONCE, on the final checkpoints, so --confirm-final is required.

Example (Kaggle, Internet off is fine; no weights are downloaded):
  python eval_test.py \
    --color-dir /kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color \
    --full-split-csv /kaggle/input/datasets/allumalla07/mobilenetv3-test-split/split.csv \
    --confirm-final
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from train import TestImages, predict   # train.py applies the Triton guard before importing torch

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from config import NUM_CLASSES, VALIDATION_BATCH_SIZE
from dataset import normalize_relative_path

p = argparse.ArgumentParser()
p.add_argument("--color-dir", required=True)
p.add_argument("--full-split-csv", required=True)
p.add_argument("--ckpt-root", default="results/mobilenetv3")
p.add_argument("--recipe", default="uniform_p010")
p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
p.add_argument("--class-map", default="class_to_idx.json")
p.add_argument("--train-val-csv", default="train_val_split.csv")
p.add_argument("--confirm-final", action="store_true",
               help="Required: confirms these are the final checkpoints and the test set is used once.")
a = p.parse_args()
if not a.confirm_final:
    p.error("Add --confirm-final (test set is single-use).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_to_idx = json.load(open(a.class_map))

full = pd.read_csv(a.full_split_csv)
full["split"] = full["split"].astype(str).str.strip().str.lower()
test_df = full[full["split"] == "test"].reset_index(drop=True)
test_df["relative_path"] = test_df["path"].map(normalize_relative_path)
tv = pd.read_csv(a.train_val_csv)
tv_paths = set(tv["path"].map(normalize_relative_path))
assert not (tv_paths & set(test_df["relative_path"])), "Leakage: test overlaps train/val"
labels = test_df["class"].astype(str).map(class_to_idx).to_numpy()
print(f"Test rows: {len(test_df)}; classes: {test_df['class'].nunique()}")

loader = DataLoader(TestImages(test_df["relative_path"], labels, a.color_dir),
                    batch_size=VALIDATION_BATCH_SIZE, shuffle=False, num_workers=2)

rows = []
for seed in a.seeds:
    run_dir = Path(a.ckpt_root) / f"mobilenetv3_large_{a.recipe}_seed{seed}"
    model = mobilenet_v3_large(weights=None)   # no download; checkpoint supplies the weights
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(device)
    model.load_state_dict(torch.load(run_dir / "best_weights.pt", map_location=device))
    y_true, y_pred = predict(model, loader, device)
    f1 = float(f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    pd.DataFrame({"path": test_df["path"], "true": y_true, "pred": y_pred}).to_csv(
        run_dir / "test_predictions.csv", index=False)
    (run_dir / "test_metrics.json").write_text(json.dumps(
        {"seed": seed, "recipe": a.recipe, "test_rows": int(len(y_true)),
         "test_macro_f1": f1, "test_accuracy": acc}, indent=2))
    print(f"seed={seed} test macro-F1={f1:.6f} acc={acc:.6f}")
    rows.append((f1, acc))

f1s = np.array([r[0] for r in rows]); accs = np.array([r[1] for r in rows])
print(f"MEAN test macro-F1 = {f1s.mean():.6f} +/- {f1s.std(ddof=1):.6f}   mean acc = {accs.mean():.6f}")
