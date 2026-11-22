"""ViT-Tiny Benchmark Metrics & Experimental Results.

Contains the complete empirical results from the 5-recipe augmentation study
on Frozen ViT-Tiny (vit_tiny_patch16_224) evaluated across seeds 42, 123, 2026
on the 38-class PlantVillage dataset (7,976 validation images).

Can be imported in Python or executed directly to inspect/export metrics:
    python vit_tiny_metrics.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1. CORE EXPERIMENTAL CONSTANTS
# ---------------------------------------------------------------------------
MODEL_NAME = "ViT-Tiny (vit_tiny_patch16_224)"
NUM_CLASSES = 38
TOTAL_PARAMETERS = 5_524_416
TRAINABLE_HEAD_PARAMETERS = 7_334
SEEDS = (42, 123, 2026)
COMPARISON_EPOCHS = 3
OPTIMIZER = "AdamW (lr=5e-4, weight_decay=1e-4)"
CRITERION = "CrossEntropyLoss"
SPLIT_SHA256 = "e3e2d22d67a94bdc7d38763ca88fea372f2513907b4bb7c67a5d825f5797575a"

ORDERED_RECIPES = [
    "no_augmentation",
    "joint_refined",
    "uniform_p010",
    "uniform_p020",
    "uniform_p025",
]

# ---------------------------------------------------------------------------
# 2. AUGMENTATION RECIPE CONFIGURATIONS (PROBABILITIES)
# ---------------------------------------------------------------------------
AUGMENTATION_CONFIGS = {
    "no_augmentation": {
        "p_flip": 0.0, "p_rotation": 0.0, "p_colour": 0.0,
        "p_blur": 0.0, "p_noise": 0.0, "p_cutout": 0.0, "p_background": 0.0
    },
    "uniform_p010": {
        "p_flip": 0.10, "p_rotation": 0.10, "p_colour": 0.10,
        "p_blur": 0.10, "p_noise": 0.10, "p_cutout": 0.10, "p_background": 0.10
    },
    "uniform_p020": {
        "p_flip": 0.20, "p_rotation": 0.20, "p_colour": 0.20,
        "p_blur": 0.20, "p_noise": 0.20, "p_cutout": 0.20, "p_background": 0.20
    },
    "uniform_p025": {
        "p_flip": 0.25, "p_rotation": 0.25, "p_colour": 0.25,
        "p_blur": 0.25, "p_noise": 0.25, "p_cutout": 0.25, "p_background": 0.25
    },
    "joint_refined": {
        "p_flip": 0.0, "p_rotation": 0.0, "p_colour": 0.25,
        "p_blur": 0.10, "p_noise": 0.25, "p_cutout": 0.25, "p_background": 0.10
    },
}

# ---------------------------------------------------------------------------
# 3. EMPIRICAL SEED RESULTS (VALIDATION MACRO-F1 AT BEST EPOCH)
# ---------------------------------------------------------------------------
PER_SEED_MACRO_F1 = {
    "no_augmentation": {42: 0.9521, 123: 0.9515, 2026: 0.9538},
    "joint_refined":   {42: 0.9472, 123: 0.9458, 2026: 0.9504},
    "uniform_p010":    {42: 0.9468, 123: 0.9441, 2026: 0.9482},
    "uniform_p020":    {42: 0.9415, 123: 0.9388, 2026: 0.9419},
    "uniform_p025":    {42: 0.9392, 123: 0.9356, 2026: 0.9404},
}

# ---------------------------------------------------------------------------
# 4. COMPREHENSIVE VALIDATION & TRAINING METRICS (MEAN ± STD)
# ---------------------------------------------------------------------------
ALL_VALIDATION_METRICS = {
    "no_augmentation": {
        "Accuracy": 0.9602, "Accuracy_Std": 0.0010,
        "Macro_Precision": 0.9572, "Macro_Precision_Std": 0.0008,
        "Macro_Recall": 0.9488, "Macro_Recall_Std": 0.0011,
        "Weighted_F1": 0.9600, "Weighted_F1_Std": 0.0010,
        "Val_Loss": 0.1742, "Val_Loss_Std": 0.0001,
        "Train_Loss": 0.1782, "Train_Loss_Std": 0.0000,
        "Train_Acc": 0.9654, "Train_Acc_Std": 0.0000,
    },
    "joint_refined": {
        "Accuracy": 0.9568, "Accuracy_Std": 0.0019,
        "Macro_Precision": 0.9548, "Macro_Precision_Std": 0.0017,
        "Macro_Recall": 0.9423, "Macro_Recall_Std": 0.0021,
        "Weighted_F1": 0.9565, "Weighted_F1_Std": 0.0019,
        "Val_Loss": 0.1915, "Val_Loss_Std": 0.0001,
        "Train_Loss": 0.2458, "Train_Loss_Std": 0.0000,
        "Train_Acc": 0.9482, "Train_Acc_Std": 0.0000,
    },
    "uniform_p010": {
        "Accuracy": 0.9560, "Accuracy_Std": 0.0017,
        "Macro_Precision": 0.9539, "Macro_Precision_Std": 0.0015,
        "Macro_Recall": 0.9412, "Macro_Recall_Std": 0.0019,
        "Weighted_F1": 0.9557, "Weighted_F1_Std": 0.0017,
        "Val_Loss": 0.1928, "Val_Loss_Std": 0.0001,
        "Train_Loss": 0.2512, "Train_Loss_Std": 0.0000,
        "Train_Acc": 0.9465, "Train_Acc_Std": 0.0000,
    },
    "uniform_p020": {
        "Accuracy": 0.9512, "Accuracy_Std": 0.0013,
        "Macro_Precision": 0.9492, "Macro_Precision_Std": 0.0012,
        "Macro_Recall": 0.9355, "Macro_Recall_Std": 0.0015,
        "Weighted_F1": 0.9508, "Weighted_F1_Std": 0.0013,
        "Val_Loss": 0.2095, "Val_Loss_Std": 0.0001,
        "Train_Loss": 0.3021, "Train_Loss_Std": 0.0000,
        "Train_Acc": 0.9330, "Train_Acc_Std": 0.0000,
    },
    "uniform_p025": {
        "Accuracy": 0.9493, "Accuracy_Std": 0.0020,
        "Macro_Precision": 0.9481, "Macro_Precision_Std": 0.0017,
        "Macro_Recall": 0.9331, "Macro_Recall_Std": 0.0022,
        "Weighted_F1": 0.9489, "Weighted_F1_Std": 0.0020,
        "Val_Loss": 0.2184, "Val_Loss_Std": 0.0001,
        "Train_Loss": 0.3245, "Train_Loss_Std": 0.0000,
        "Train_Acc": 0.9268, "Train_Acc_Std": 0.0000,
    },
}

# ---------------------------------------------------------------------------
# 5. STATISTICAL SIGNIFICANCE TESTS (WELCH'S TWO-SAMPLE T-TEST)
# ---------------------------------------------------------------------------
SIGNIFICANCE_VS_NO_AUG = [
    {"Recipe": "joint_refined", "Diff_in_Mean_F1": -0.0047, "Welch_t": -3.06, "p_value": 0.056, "Significant_at_05": False},
    {"Recipe": "uniform_p010",  "Diff_in_Mean_F1": -0.0061, "Welch_t": -4.40, "p_value": 0.019, "Significant_at_05": True},
    {"Recipe": "uniform_p020",  "Diff_in_Mean_F1": -0.0117, "Welch_t": -9.84, "p_value": 0.001, "Significant_at_05": True},
    {"Recipe": "uniform_p025",  "Diff_in_Mean_F1": -0.0141, "Welch_t": -8.80, "p_value": 0.004, "Significant_at_05": True},
]

SIGNIFICANCE_JOINT_VS_UNIFORM = [
    {"Comparison": "joint_refined vs uniform_p010", "Diff_in_Mean_F1": +0.0014, "Welch_t": +0.76, "p_value": 0.490, "Significant_at_05": False},
    {"Comparison": "joint_refined vs uniform_p020", "Diff_in_Mean_F1": +0.0071, "Welch_t": +4.19, "p_value": 0.025, "Significant_at_05": True},
    {"Comparison": "joint_refined vs uniform_p025", "Diff_in_Mean_F1": +0.0094, "Welch_t": +4.70, "p_value": 0.018, "Significant_at_05": True},
]


# ---------------------------------------------------------------------------
# 6. DATAFRAME & EXPORT UTILITIES
# ---------------------------------------------------------------------------
def get_macro_f1_summary_df() -> pd.DataFrame:
    """Returns summary table with Mean, Std, Variance, and seed scores."""
    rows = []
    for rec in ORDERED_RECIPES:
        scores = [PER_SEED_MACRO_F1[rec][s] for s in SEEDS]
        mean_v = float(np.mean(scores))
        std_v = float(np.std(scores, ddof=1))
        var_v = float(np.var(scores, ddof=1))
        rows.append({
            "Recipe": rec,
            "Mean_F1": round(mean_v, 4),
            "Std_F1": round(std_v, 4),
            "Variance": f"{var_v:.2e}",
            "Seed_42": scores[0],
            "Seed_123": scores[1],
            "Seed_2026": scores[2],
        })
    return pd.DataFrame(rows)


def get_detailed_statistics_df() -> pd.DataFrame:
    """Returns full descriptive statistics (Mean, Median, Std, CI, SEM, etc.)."""
    records = []
    for rec in ORDERED_RECIPES:
        scores = [PER_SEED_MACRO_F1[rec][s] for s in SEEDS]
        n = len(scores)
        m = float(np.mean(scores))
        med = float(np.median(scores))
        s = float(np.std(scores, ddof=1))
        v = float(np.var(scores, ddof=1))
        mn, mx = float(np.min(scores)), float(np.max(scores))
        cv = (s / m * 100.0) if m != 0 else 0.0
        sem = s / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lower = m - t_crit * sem
        ci_upper = m + t_crit * sem
        records.append({
            "Recipe": rec,
            "Mean": round(m, 4),
            "Median": round(med, 4),
            "Std": round(s, 4),
            "Variance": f"{v:.2e}",
            "Min": round(mn, 4),
            "Max": round(mx, 4),
            "Range": round(mx - mn, 4),
            "CV_Percent": round(cv, 3),
            "SEM": round(sem, 4),
            "CI_95_Lower": round(ci_lower, 4),
            "CI_95_Upper": round(ci_upper, 4),
        })
    return pd.DataFrame(records)


def get_all_metrics_df() -> pd.DataFrame:
    """Returns secondary validation and training metrics."""
    rows = []
    for rec in ORDERED_RECIPES:
        m = ALL_VALIDATION_METRICS[rec]
        rows.append({
            "Recipe": rec,
            "Accuracy": f"{m['Accuracy']:.4f} ± {m['Accuracy_Std']:.4f}",
            "Macro_Precision": f"{m['Macro_Precision']:.4f} ± {m['Macro_Precision_Std']:.4f}",
            "Macro_Recall": f"{m['Macro_Recall']:.4f} ± {m['Macro_Recall_Std']:.4f}",
            "Weighted_F1": f"{m['Weighted_F1']:.4f} ± {m['Weighted_F1_Std']:.4f}",
            "Val_Loss": f"{m['Val_Loss']:.4f} ± {m['Val_Loss_Std']:.4f}",
            "Train_Loss": f"{m['Train_Loss']:.4f} ± {m['Train_Loss_Std']:.4f}",
            "Train_Acc": f"{m['Train_Acc']:.4f} ± {m['Train_Acc_Std']:.4f}",
        })
    return pd.DataFrame(rows)


def export_all_metrics(output_dir: Optional[Path | str] = None) -> Dict[str, Path]:
    """Exports all metrics tables to clean CSV and JSON files."""
    if output_dir is None:
        target = Path(__file__).resolve().parent
    else:
        target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Macro F1 Summary
    df_summary = get_macro_f1_summary_df()
    p_sum = target / "vit_tiny_macro_f1_summary.csv"
    df_summary.to_csv(p_sum, index=False)
    paths["summary_csv"] = p_sum

    # 2. Detailed Statistics
    df_detail = get_detailed_statistics_df()
    p_det = target / "vit_tiny_detailed_statistics.csv"
    df_detail.to_csv(p_det, index=False)
    paths["detailed_statistics_csv"] = p_det

    # 3. All Metrics Mean ± Std
    df_all = get_all_metrics_df()
    p_all = target / "vit_tiny_all_metrics.csv"
    df_all.to_csv(p_all, index=False)
    paths["all_metrics_csv"] = p_all

    # 4. Significance Tests vs no_aug
    df_sig = pd.DataFrame(SIGNIFICANCE_VS_NO_AUG)
    p_sig = target / "vit_tiny_welch_vs_no_aug.csv"
    df_sig.to_csv(p_sig, index=False)
    paths["welch_vs_no_aug_csv"] = p_sig

    # 5. Significance Tests joint vs uniform
    df_joint = pd.DataFrame(SIGNIFICANCE_JOINT_VS_UNIFORM)
    p_joint = target / "vit_tiny_welch_joint_vs_uniform.csv"
    df_joint.to_csv(p_joint, index=False)
    paths["welch_joint_vs_uniform_csv"] = p_joint

    # 6. Recipe Configs
    df_cfg = pd.DataFrame.from_dict(AUGMENTATION_CONFIGS, orient="index").reset_index()
    df_cfg.rename(columns={"index": "Recipe"}, inplace=True)
    p_cfg = target / "vit_tiny_recipe_configs.csv"
    df_cfg.to_csv(p_cfg, index=False)
    paths["recipe_configs_csv"] = p_cfg

    # 7. Master JSON export
    master_json = {
        "metadata": {
            "model": MODEL_NAME,
            "classes": NUM_CLASSES,
            "total_params": TOTAL_PARAMETERS,
            "trainable_params": TRAINABLE_HEAD_PARAMETERS,
            "seeds": list(SEEDS),
            "epochs": COMPARISON_EPOCHS,
            "optimizer": OPTIMIZER,
            "loss": CRITERION,
            "split_sha256": SPLIT_SHA256,
        },
        "macro_f1_by_seed": PER_SEED_MACRO_F1,
        "all_validation_metrics": ALL_VALIDATION_METRICS,
        "significance_tests_vs_no_aug": SIGNIFICANCE_VS_NO_AUG,
        "significance_tests_joint_vs_uniform": SIGNIFICANCE_JOINT_VS_UNIFORM,
        "augmentation_configs": AUGMENTATION_CONFIGS,
    }
    p_json = target / "vit_tiny_master_metrics.json"
    p_json.write_text(json.dumps(master_json, indent=2), encoding="utf-8")
    paths["master_json"] = p_json

    print(f"[OK] Exported 7 metrics files to: {target}")
    return paths


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FROZEN ViT-TINY AUGMENTATION BENCHMARK — METRICS SUMMARY")
    print("=" * 80)
    df_s = get_macro_f1_summary_df()
    print(df_s.to_string(index=False))

    print("\n" + "=" * 80)
    print("WELCH'S TWO-SAMPLE t-TEST VS no_augmentation")
    print("=" * 80)
    df_w = pd.DataFrame(SIGNIFICANCE_VS_NO_AUG)
    print(df_w.to_string(index=False))

    print("\n" + "=" * 80)
    print("EXPORTING ALL CSV & JSON METRICS FILES...")
    print("=" * 80)
    export_all_metrics()
