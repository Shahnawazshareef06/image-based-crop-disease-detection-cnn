"""Benchmark ViT-Tiny model size, parameter counts, latency, and throughput."""

from __future__ import annotations

import json
from pathlib import Path
import torch

from model import build_vit_tiny, count_parameters, benchmark_inference
from config import NUM_CLASSES


def run_benchmark() -> dict:
    device = torch.device("cpu")
    print(f"Benchmarking ViT-Tiny on {device}...")

    model = build_vit_tiny(num_classes=NUM_CLASSES, pretrained=False)
    params = count_parameters(model)

    # Benchmark checkpoint size
    temp_ckpt = Path("temp_vit_tiny_benchmark.pth")
    torch.save({"model_state_dict": model.state_dict()}, temp_ckpt)
    ckpt_size_mb = temp_ckpt.stat().st_size / (1024 * 1024)
    temp_ckpt.unlink(missing_ok=True)

    perf = benchmark_inference(
        model=model,
        device=device,
        batch_size=32,
        num_warmup=10,
        num_runs=30,
    )

    results = {
        "model_architecture": "vit_tiny_patch16_224",
        "library": "timm",
        "num_classes": NUM_CLASSES,
        "parameters": params,
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "performance": perf,
    }

    print("\nBenchmark Results:")
    print(json.dumps(results, indent=2))

    with open("vit_tiny_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_benchmark()
