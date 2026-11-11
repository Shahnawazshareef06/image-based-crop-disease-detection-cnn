"""ViT-Tiny model definition, parameter grouping, and benchmarking utilities."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import timm

from config import (
    BACKBONE_LR,
    CLASSIFIER_LR,
    IMAGE_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED,
    WEIGHT_DECAY,
)


def build_vit_tiny(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
) -> nn.Module:
    """Instantiate vit_tiny_patch16_224 with a fresh classification head.

    Replaces the ImageNet-1K pretrained head with a new Linear(in_features, num_classes).
    All backbone and head parameters are trainable (full fine-tuning).
    """
    # Create ViT-Tiny model from timm
    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Explicitly ensure head has in_features -> num_classes and correct initialization
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        if model.head.out_features != num_classes:
            model.head = nn.Linear(in_features, num_classes)
            nn.init.trunc_normal_(model.head.weight, std=0.02)
            nn.init.zeros_(model.head.bias)
    elif hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)

    return model


def get_optimizer_param_groups(
    model: nn.Module,
    backbone_lr: float = BACKBONE_LR,
    classifier_lr: float = CLASSIFIER_LR,
    weight_decay: float = WEIGHT_DECAY,
) -> list[dict[str, Any]]:
    """Split model parameters into separate backbone and classifier parameter groups.

    Backbone group uses backbone_lr (1e-5).
    Classifier group (head) uses classifier_lr (5e-4).
    Both use weight_decay (1e-4).
    """
    backbone_params: list[torch.nn.Parameter] = []
    classifier_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Classifier head in ViT-Tiny timm is model.head
        if name.startswith("head.") or name.startswith("classifier."):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone",
        },
        {
            "params": classifier_params,
            "lr": classifier_lr,
            "weight_decay": weight_decay,
            "name": "classifier",
        },
    ]


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classifier_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if (name.startswith("head.") or name.startswith("classifier.")) and p.requires_grad
    )
    backbone_params = trainable_params - classifier_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbone_parameters": backbone_params,
        "classifier_parameters": classifier_params,
    }


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    num_warmup: int = 10,
    num_runs: int = 50,
    input_size: tuple[int, int, int] = (3, IMAGE_SIZE, IMAGE_SIZE),
) -> dict[str, Any]:
    """Measure inference latency (single image) and throughput (batched)."""
    model.eval()
    model.to(device)

    # 1. Single image latency
    single_input = torch.randn(1, *input_size, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(single_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(single_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        single_latency_ms = ((time.perf_counter() - start_time) / num_runs) * 1000.0

    # 2. Batched throughput
    batched_input = torch.randn(batch_size, *input_size, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(batched_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(batched_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        total_images = num_runs * batch_size
        throughput_img_per_sec = total_images / elapsed
        batched_latency_ms = (elapsed / num_runs) * 1000.0

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "single_image_latency_ms": round(single_latency_ms, 2),
        "batch_size": batch_size,
        "batched_latency_ms": round(batched_latency_ms, 2),
        "throughput_images_per_sec": round(throughput_img_per_sec, 2),
        "warmup_iterations": num_warmup,
        "benchmark_runs": num_runs,
    }
