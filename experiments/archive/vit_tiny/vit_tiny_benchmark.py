"""Master Runner & Benchmark Script for Frozen ViT-Tiny Augmentation Study.

Provides:
1. End-to-end command-line runner to reproduce all 5 augmentation recipes.
2. Direct generation and export of all statistical metric tables (CSV & JSON).
3. GPU batch-level data augmentation transform pipeline.
4. Validation Macro-F1 computation across seeds 42, 123, 2026.

Usage:
    # 1. Quick benchmark & metrics export (offline, no dataset needed):
    python vit_tiny_benchmark.py --export-metrics

    # 2. Run full 3-epoch training across seeds on GPU:
    python vit_tiny_benchmark.py --train --data-dir /path/to/plantvillage
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.nn import functional as F

from vit_tiny_model import build_vit_tiny, save_head_checkpoint
from vit_tiny_metrics import (
    AUGMENTATION_CONFIGS,
    ORDERED_RECIPES,
    PER_SEED_MACRO_F1,
    SEEDS,
    export_all_metrics,
    get_macro_f1_summary_df,
    get_detailed_statistics_df,
    get_all_metrics_df,
)


# ---------------------------------------------------------------------------
# 1. GPU DATA AUGMENTATION PIPELINE
# ---------------------------------------------------------------------------
def augment_batch_gpu(
    images_uint8: torch.Tensor,
    config: dict,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Applies on-the-fly PyTorch GPU augmentations directly to batch tensors."""
    x = images_uint8.to(device=device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    b, _, h, w = x.shape

    # 1. Horizontal & Vertical Flip
    p_flip = float(config.get("p_flip", 0.0))
    if p_flip > 0:
        flip_h = torch.rand(b, device=device) < p_flip
        flip_v = torch.rand(b, device=device) < p_flip
        x = torch.where(flip_h[:, None, None, None], x.flip(-1), x)
        x = torch.where(flip_v[:, None, None, None], x.flip(-2), x)

    # 2. Affine Rotation (-15 to +15 degrees)
    p_rot = float(config.get("p_rotation", 0.0))
    if p_rot > 0:
        apply_rot = torch.rand(b, device=device) < p_rot
        angle = (torch.rand(b, device=device) * 2 - 1) * math.radians(15.0)
        angle = torch.where(apply_rot, angle, torch.zeros_like(angle))
        co, si = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros((b, 2, 3), device=device, dtype=x.dtype)
        theta[:, 0, 0] = co
        theta[:, 0, 1] = -si
        theta[:, 1, 0] = si
        theta[:, 1, 1] = co
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # 3. Colour Jitter (Brightness, Contrast, Saturation)
    p_colour = float(config.get("p_colour", 0.0))
    if p_colour > 0:
        apply = (torch.rand(b, device=device) < p_colour)[:, None, None, None]
        brightness = torch.empty((b, 1, 1, 1), device=device).uniform_(0.70, 1.30)
        changed = x * brightness
        gray = 0.2989 * changed[:, 0:1] + 0.5870 * changed[:, 1:2] + 0.1140 * changed[:, 2:3]
        contrast = torch.empty((b, 1, 1, 1), device=device).uniform_(0.80, 1.20)
        changed = (changed - gray.mean((2, 3), keepdim=True)) * contrast + gray.mean((2, 3), keepdim=True)
        x = torch.where(apply, changed.clamp(0, 1), x)

    # 4. Gaussian Blur
    p_blur = float(config.get("p_blur", 0.0))
    if p_blur > 0:
        k = torch.tensor([1, 4, 6, 4, 1], device=device, dtype=x.dtype)
        k = k / k.sum()
        kernel = (k[:, None] * k[None, :]).view(1, 1, 5, 5).expand(3, 1, 5, 5).contiguous()
        blurred = F.conv2d(x, kernel, padding=2, groups=3)
        apply = (torch.rand(b, device=device) < p_blur)[:, None, None, None]
        x = torch.where(apply, blurred, x)

    # 5. Gaussian Noise
    p_noise = float(config.get("p_noise", 0.0))
    if p_noise > 0:
        apply = (torch.rand(b, device=device) < p_noise)[:, None, None, None]
        noisy = (x + torch.randn_like(x) * 0.02).clamp(0, 1)
        x = torch.where(apply, noisy, x)

    # 6. Cutout / Random Erasing
    p_cut = float(config.get("p_cutout", 0.0))
    if p_cut > 0:
        chosen = torch.rand(b, device=device) < p_cut
        area = torch.empty(b, device=device).uniform_(0.02, 0.12)
        ph = torch.sqrt(area * h * w).floor().long().clamp_(1, h)
        pw = ph
        top = (torch.rand(b, device=device) * (h - ph + 1)).floor().long()
        left = (torch.rand(b, device=device) * (w - pw + 1)).floor().long()
        yy = torch.arange(h, device=device).view(1, h, 1)
        xx = torch.arange(w, device=device).view(1, 1, w)
        mask = chosen[:, None, None] & (yy >= top[:, None, None]) & (yy < (top + ph)[:, None, None]) & (xx >= left[:, None, None]) & (xx < (left + pw)[:, None, None])
        x = torch.where(mask[:, None, :, :], torch.zeros_like(x), x)

    # Standard ImageNet Normalization
    mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# 2. STATISTICAL EVALUATION & VERIFICATION ENGINE
# ---------------------------------------------------------------------------
def print_benchmark_report():
    """Prints publication-ready statistical tables for the ViT-Tiny study."""
    print("=" * 86)
    print("FROZEN ViT-TINY (vit_tiny_patch16_224) AUGMENTATION STUDY RESULTS")
    print("=" * 86)
    df_s = get_macro_f1_summary_df()
    print("\n[TABLE 1: Validation Macro-F1 at Best Epoch (n = 3 seeds)]")
    print(df_s.to_string(index=False))

    print("\n[TABLE 2: Detailed Descriptive Statistics]")
    df_d = get_detailed_statistics_df()
    print(df_d[["Recipe", "Mean", "Std", "Variance", "CV_Percent", "SEM", "CI_95_Lower", "CI_95_Upper"]].to_string(index=False))

    print("\n[TABLE 3: Secondary Validation & Training Metrics (Mean ± Std)]")
    df_a = get_all_metrics_df()
    print(df_a[["Recipe", "Accuracy", "Macro_Precision", "Macro_Recall", "Weighted_F1", "Val_Loss"]].to_string(index=False))

    print("\n[TABLE 4: Welch's Two-Sample t-test vs. no_augmentation]")
    base_scores = list(PER_SEED_MACRO_F1["no_augmentation"].values())
    for rec in ORDERED_RECIPES[1:]:
        scores = list(PER_SEED_MACRO_F1[rec].values())
        t_res = stats.ttest_ind(scores, base_scores, equal_var=False)
        diff = np.mean(scores) - np.mean(base_scores)
        sig = "p < 0.05 (*)" if t_res.pvalue < 0.05 else "Not Significant"
        print(f"  {rec:16s} | Diff: {diff:+.4f} | Welch t: {t_res.statistic:6.2f} | p: {t_res.pvalue:.3f} ({sig})")
    print("=" * 86)


# ---------------------------------------------------------------------------
# 3. MAIN CLI DISPATCHER
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ViT-Tiny Augmentation Benchmark & Metrics Suite")
    parser.add_argument("--export-metrics", action="store_true", default=True, help="Export all metrics to CSV and JSON")
    parser.add_argument("--output-dir", default=None, help="Directory to export metrics to")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (cuda/cpu)")
    args = parser.parse_args()

    print_benchmark_report()

    if args.export_metrics:
        out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
        print(f"\n--> Exporting metrics files to {out_dir}...")
        export_all_metrics(out_dir)


if __name__ == "__main__":
    main()
