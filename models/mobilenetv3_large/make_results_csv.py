"""Build results/mean_std_results.csv (same columns as the ViT-Tiny handover)
from the metrics.json files written by train.py.

Usage: python make_results_csv.py <results_dir> [<results_dir> ...]
e.g.   python make_results_csv.py /kaggle/working/results/mobilenetv3 /kaggle/working/results/mobilenetv3_aug
"""
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = (42, 123, 2026)
by_recipe = defaultdict(dict)
for d in sys.argv[1:]:
    for f in glob.glob(f"{d}/mobilenetv3_*_seed*/metrics.json"):
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        by_recipe[m["recipe"]][m["seed"]] = (
            m["best_validation_macro_f1"], m["best_validation_metrics"]["val_accuracy"])
assert by_recipe, "No metrics.json found - run train.py first."

rows = []
for recipe, runs in by_recipe.items():
    f1 = np.array([v[0] for v in runs.values()]); acc = np.array([v[1] for v in runs.values()])
    rows.append((recipe, runs, f1.mean(), f1.std(ddof=1) if len(f1) > 1 else 0.0, acc.mean()))
rows.sort(key=lambda r: -r[2])

out = Path("results"); out.mkdir(exist_ok=True)
with open(out / "mean_std_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["rank", "recipe_name", "seed_42_macro_f1", "seed_123_macro_f1", "seed_2026_macro_f1",
                "mean_macro_f1", "std_macro_f1", "mean_accuracy"])
    for i, (recipe, runs, mean, std, acc) in enumerate(rows, 1):
        w.writerow([i, recipe] + [round(runs[s][0], 4) if s in runs else "" for s in SEEDS]
                   + [round(mean, 6), round(std, 6), round(acc, 4)])
print(open(out / "mean_std_results.csv").read())
