"""Mean and sample standard deviation of validation macro-F1 across seeds."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/mobilenetv3"
files = sorted(glob.glob(f"{run_dir}/mobilenetv3_*_seed*/metrics.json"))
runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
assert runs, f"No metrics.json files found in {run_dir}"


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


f1_m, f1_s = mean_sd([r["best_validation_macro_f1"] for r in runs])
acc_m, acc_s = mean_sd([r["best_validation_metrics"]["val_accuracy"] for r in runs])
print(f"Model        : {runs[0]['model']} | recipe: {runs[0]['recipe']}")
print(f"Seeds        : {[r['seed'] for r in runs]}  (best epochs: {[r['best_epoch'] for r in runs]})")
print(f"Val macro-F1 : {f1_m:.4f} +/- {f1_s:.4f}")
print(f"Val accuracy : {acc_m:.4f} +/- {acc_s:.4f}")
print(f"Parameters   : {runs[0]['parameters_total']:,}")
print(f"Weights file : {runs[0]['weights_file_size_mb']} MB")
print(f"GPU time/img : {np.mean([r['gpu_ms_per_image_bs64_fp16'] for r in runs]):.4f} ms (bs64, fp16)")
print(f"CPU time/img : {np.mean([r['cpu_ms_per_image_bs1_fp32_1thread'] for r in runs]):.3f} ms (bs1, 1 thread)")